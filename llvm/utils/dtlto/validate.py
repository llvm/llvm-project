"""
DTLTO JSON Validator.

This script is used for DTLTO testing to check that the distributor has
been invoked correctly.

Usage:
    python validate.py <json_file>

Arguments:
    - <json_file> : JSON file describing the DTLTO jobs.

The script does the following:
    1. Prints the supplied distributor arguments.
    2. Loads the JSON file.
    3. Pretty prints the JSON.
    4. Validates the structure and required fields.
"""

import sys
import json
from pathlib import Path


def take(jvalue, jpath):
    parts = jpath.split(".")
    for part in parts[:-1]:
        jvalue = jvalue[part]
    return jvalue.pop(parts[-1], KeyError)


def validate(jdoc):
    # Check the format of the JSON
    assert type(take(jdoc, "common.linker_output")) is str

    args = take(jdoc, "common.args")
    assert type(args) is list
    assert len(args) > 0
    assert all(type(i) is str for i in args)

    inputs = take(jdoc, "common.inputs")
    assert type(inputs) is list
    assert all(type(i) is str for i in inputs)

    assert len(take(jdoc, "common")) == 0

    jobs = take(jdoc, "jobs")
    assert type(jobs) is list
    for j in jobs:
        assert type(j) is dict

        for attr, min_size in (("args", 0), ("inputs", 2), ("outputs", 1)):
            array = take(j, attr)
            assert len(array) >= min_size
            assert type(array) is list
            assert all(type(a) is str for a in array)

        assert len(j) == 0

    assert len(jdoc) == 0


if __name__ == "__main__":
    json_arg = Path(sys.argv[-1])
    distributor_args = sys.argv[1:-1]

    # Print the supplied distributor arguments.
    print(f"{distributor_args=}")

    # Load the DTLTO information from the input JSON file.
    with json_arg.open() as f:
        jdoc = json.load(f)

    # Write the input JSON to stdout.
    print(json.dumps(jdoc, indent=4))

    # Check the format of the JSON.
    validate(jdoc)
