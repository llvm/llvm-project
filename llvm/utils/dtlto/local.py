"""
DTLTO local serial distributor.

This script parses the Distributed ThinLTO (DTLTO) JSON file and serially
executes the specified code generation tool on the local host to perform each 
backend compilation job. This simple functional distributor is intended to be
used for integration tests.

Usage:
    python dtlto_codegen.py <json_file>

Arguments:
    - <json_file> : JSON file describing the DTLTO jobs.
"""

import subprocess
import sys
import json
from pathlib import Path

if __name__ == "__main__":
    # Load the DTLTO information from the input JSON file.
    with Path(sys.argv[-1]).open() as f:
        data = json.load(f)

    # Iterate over the jobs and execute the codegen tool.
    for job in data["jobs"]:
        jobargs = []
        for arg in data["common"]["args"]:
            if isinstance(arg, list):
                # arg is a "template", into which an external filename is to be
                # inserted. The first element of arg names an array of strings
                # in the job. The remaining elements of arg are either indices
                # into the array or literal strings.
                files, rest = job[arg[0]], arg[1:]
                jobargs.append(
                    "".join(files[x] if isinstance(x, int) else x for x in rest)
                )
            else:
                jobargs.append(arg)
        subprocess.check_call(jobargs)
