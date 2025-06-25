"""
Copyright (c) 2022

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List, Optional
from pathlib import Path
import os
import subprocess
import json
import logging
import pypremise

from pypremise.core import PremiseResult, Direction, Pattern, PremiseInstance

PREMISE_CONFIG_FILENAME = ".pypremise_config.json"
PREMISE_ENGINE_NAME_LINUX = "Linux"
PREMISE_ENGINE_NAME_APPLE_SILICON = "AppleSilicon"
PREMISE_LINUX_FILENAME = "Premise_Linux"
PREMISE_APPLESILICON_FILENAME = "Premise_AppleSilicon"

logger = logging.getLogger(__name__)

def parse_premise_result(path: str, group_0_name: str, group_1_name: str) -> List[PremiseResult]:
    results = []
    with open(path, "r") as in_file:
        for line in in_file:
            tail, counts, bias_direction, fisher, mdl_gain, _ = line.strip().split(";")
            fisher = float(fisher)
            mdl_gain = float(mdl_gain)
            clauses = []
            for clause in tail.split(" and "):
                clause_tokens = []
                for item in clause.split("-or-"):
                    clause_tokens.append(int(item))
                clauses.append(clause_tokens)
            label0_count, label1_count = counts.strip().split(":")
            label0_count = int(label0_count.strip())
            label1_count = int(label1_count.strip())
            direction = Direction.GROUP_0 if bias_direction == "0" else Direction.GROUP_1
            pattern = Pattern(clauses)
            results.append(PremiseResult(pattern, direction, label0_count, label1_count, fisher, mdl_gain, group_0_name, group_1_name))
    return results


def write_dat_content(instances: List[PremiseInstance], feature_path: str, label_path: str) -> None:
    feature_file = open(feature_path, "w")
    label_file = open(label_path, "w")
    for instance in instances:
        feature_file.write(" ".join([str(f) for f in instance.features]) + "\n")
        label_file.write(f"{instance.label.value}\n")
    feature_file.flush()
    label_file.flush()


def write_embedding_file(embedding_index_to_vector, embedding_path: str, embedding_dimensionality: int,
                         max_feature_index: int) -> None:
    out_file = open(embedding_path, "w")
    for index in range(max_feature_index+1):
            if index not in embedding_index_to_vector:
                raise Exception(f"Index {index} could not be found in the embedding map embedding_index_to_vector."
                                f"The map needs to contain values for all feature indices from 0 to "
                                f"max feature index ({max_feature_index}) even if this index does not appear in the "
                                f"dataset.")
            vector = embedding_index_to_vector[index]
            if len(vector) != embedding_dimensionality:
                raise Exception(f"Length of the embedding vector is {len(vector)} but the embedding dimensionality"
                                f"was specified as {embedding_dimensionality}.")
            out_file.write(" ".join([str(v) for v in vector]) + "\n")
    out_file.flush()


def call_premise_program(feature_path: str, label_path: str, result_path: str, embedding_path: str,
                         embedding_size: int, neighbor_max_distance: int, fisher_p_value: float,
                         clause_max_overlap: float, min_overlap: float, 
                         premise_engine: Optional[str] = None) -> None:
    premise_path = get_premise_path(premise_engine)

    logger.info("Starting Premise. This might take a while.")
    try:
        stdout_output = ""
        with subprocess.Popen([premise_path, feature_path, label_path, result_path,
                                    embedding_path,
                                    str(embedding_size), str(neighbor_max_distance),
                                    str(fisher_p_value), str(clause_max_overlap), str(min_overlap)],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                logger.info(line)
            stdout_output += line
            process.wait()

            stderr_output = process.stderr.read()

    except Exception as e:
        raise Exception(f"Execution of Premise failed due to '{e}'.")
    
    if len(stderr_output) > 0:
        logger.error(f"Premise reported an error.\nThe error output was" + 
                     "\n\n-------\n{stderr}\n-------")
        raise Exception(f"Premise reported an error.")


def get_premise_path(premise_engine: Optional[str] = None):
    """ The actual Premise algorithm is implemented in C++ and compiled
        to an executable to obtain resonable runtimes.
        Compiling during installation of the Python package is not yet 
        implemented. Instead, compiled versions for Linux and Mac (Apple Silicon)
        are part of the package. This method tries to detect which is the right
        version (and stores this result to avoid repeated tests in the future).
    """

    # user specified a specific local path, 
    # let's use that instead of the compiled versions that are
    # part of this package
    if premise_engine is not None and premise_engine.startswith("local_path:"):
        path = premise_engine.split(":")[1]
        logger.debug(f"Using user specified path for Premise: {path}")
        return path

    if premise_engine == None:
        # check if premise engine is specified in a config file in the home directory
        config_path = os.path.join(Path.home(), PREMISE_CONFIG_FILENAME)
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r") as in_file:
                    premise_engine = json.load(in_file).get("premise_engine", None)
                    logger.debug(f"Using Premise engine from config file {config_path}.")
            except Exception as e:
                logger.warning(f"Could not read from config path {config_path} even though " + 
                               f"config file exists due to '{e}'.")

    # the Premise executables are stored within the PyPremise module
    module_path = os.path.dirname(pypremise.__file__)

    # if premise engine not specified explicitly or via config file,
    # we have to try it out
    if premise_engine == None: 
        logger.debug("Premise engine was not defined. Trying out different engines.")
        if try_premise(os.path.join(module_path, PREMISE_LINUX_FILENAME)):
            premise_engine = PREMISE_ENGINE_NAME_LINUX
        elif try_premise(os.path.join(module_path, PREMISE_APPLESILICON_FILENAME)):
            premise_engine = PREMISE_ENGINE_NAME_APPLE_SILICON
        
        if premise_engine != None:
            # we found one! Let's try to store it
            config_path = os.path.join(Path.home(), PREMISE_CONFIG_FILENAME)
            logger.debug(f"Found a working Premise engine. Storing it in {config_path} for next time.")
            try:
                with open(config_path, "w") as out_file:
                    json.dump({"premise_engine": premise_engine}, out_file)
            except Exception as e:
                logger.warning(f"Could not store the identified Premise engine in the config file {config_path} due to '{e}'." +
                               "Next time, we'll have to go through the engine search again.")

    if premise_engine is None:
        raise Exception("Premise engine is set to 'None' and we could not identify a working " + 
                        "Premise engine for your machine. Maybe your operating system is not supported? " + 
                        "Please check with the documentation.")
    
    if premise_engine == PREMISE_ENGINE_NAME_LINUX:
        path = os.path.join(module_path, PREMISE_LINUX_FILENAME)
    elif premise_engine == PREMISE_ENGINE_NAME_APPLE_SILICON:
        path = os.path.join(module_path, PREMISE_APPLESILICON_FILENAME)
    else:
        raise Exception(f"Unknown Premise engine '{premise_engine}'")

    logging.debug(f"Using Premise engine {premise_engine} at path {path}.")
    return path

def try_premise(path):
    """
    Tries out if the given Premise executable works (defined by path) by
    runnig it with empty paratmers.
    """
    logger.debug(f"Trying out Premise executable at {path} to identify a Premise engine for your system.")
    works = True
    try:
        process = subprocess.Popen(path, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        logger.debug(f"Output of this test was: stdout='{stdout}' and stderr='{stderr}'. " + 
                     "An error output does not necessarily mean that it did not work since we run this call to the executable without parameters.")

        # TODO: Currently, Premise fails with an uncaught runtime error exception 
        # if no arguments are given (along with an error message)
        # Change to "return 0"
        # Then we could actually check the return code of the binary.
    except Exception as e: # if the binary is not suited for the user's OS, we get an OS error
       works = False 
       logger.debug(f"This executable did not work, probably because it is not compiled for your system architecture. System error: '{e}'.")
    
    return works