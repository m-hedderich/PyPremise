# Running PyPremise on Other Platforms

Premise (the base of PyPremise) is written in C++ for performance reasons and needs to be compiled for your specific platform. PyPremise comes with versions for Linux (compiled for Ubuntu 64bit) and Mac (Apple Silicon, the "M" processors). 

PyPremise should be able to automatically figure out if it can run on your system. If you run the test code

```Python
import logging
logging.basicConfig(level=logging.DEBUG)
from pypremise import Premise, data_loaders

premise_instances, _, voc_index_to_token = data_loaders.get_dummy_data()

premise = Premise(voc_index_to_token=voc_index_to_token)
patterns = premise.find_patterns(premise_instances)
```

and you get the output

```
Premise engine is set to 'None' and we could not identify a working Premise engine for your machine. Maybe your operating system is not supported? Please check with the documentation.
```

then your operating system might not be supported.

In that case, you have to first compile the [Premise](https://github.com/uds-lsv/premise/) (not PyPremise) code manually. The result is an executable Premise file.

Then you just have to tell PyPremise where to find the new executable:

```Python
premise_path = "path/to/your/new/executable"

premise = Premise(voc_index_to_token=voc_index_to_token, premise_engine="local_path:" + premise_path)
```

If you run into any issues, feel free to reach out to us.