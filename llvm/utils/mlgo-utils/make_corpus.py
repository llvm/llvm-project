#!/usr/bin/env python3

import re
import sys
from mlgo.corpus.make_corpus import parse_args_and_run

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(parse_args_and_run())
