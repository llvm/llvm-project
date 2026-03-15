"""
DTLTO local serial distributor.

This script parses the Distributed ThinLTO (DTLTO) JSON file and serially
executes the specified code generation tool on the local host to perform each 
backend compilation job. This simple functional distributor is intended to be
used for integration tests.

Usage:
    python local.py <json_file>

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
        subprocess.check_call(data["common"]["args"] + job["args"])
