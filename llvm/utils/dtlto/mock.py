"""
DTLTO Mock Distributor.

This script acts as a mock distributor for Distributed ThinLTO (DTLTO). It is
used for testing DTLTO when a Clang binary is not be available to invoke to
perform the backend compilation jobs.

Usage:
    python mock.py <input_file1> <input_file2> ... <json_file>

Arguments:
    - <input_file1>, <input_file2>, ...  : Input files to be copied.
    - <json_file>                        : JSON file describing the DTLTO jobs.

The script performs the following:
    1. Reads the JSON file containing job descriptions.
    2. For each job copies the corresponding input file to the output location
       specified for that job.
    3. Validates the JSON format using the `validate` module.
"""

import sys
import json
import shutil
from pathlib import Path
import validate

if __name__ == "__main__":
    json_arg = sys.argv[-1]
    input_files = sys.argv[1:-1]

    # Load the DTLTO information from the input JSON file.
    with Path(json_arg).open() as f:
        data = json.load(f)

    # Iterate over the jobs and create the output
    # files by copying over the supplied input files.
    for job_index, job in enumerate(data["jobs"]):
        shutil.copy(input_files[job_index], job["outputs"][0])

    # Check the format of the JSON.
    validate.validate(data)
