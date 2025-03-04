"""
DTLTO JSON Validator.

This script is used for DTLTO testing to check that the distributor has
been invoked correctly.

Usage:
    python validate.py <json_file>

Arguments:
    - <json_file> : JSON file describing the DTLTO jobs.

The script does the following:
    1. Prints the supplied CLI arguments.
    2. Loads the JSON file.
    3. Validates the structure and required fields.
    4. Pretty prints the JSON.
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

    def validate_reference(a):
        for j in jdoc["jobs"]:
            for x in a[1:]:
                if type(x) is int:
                    if a[0] not in j or x >= len(j[a[0]]):
                        return False
        return True

    for a in args:
        assert type(a) is str or (
            type(a) is list
            and len(a) >= 2
            and type(a[0]) is str
            and all(type(x) in (str, int) for x in a[1:])
            and any(type(x) is int for x in a[1:])
            and validate_reference(a)
        )

    assert len(take(jdoc, "common")) == 0

    jobs = take(jdoc, "jobs")
    assert type(jobs) is list
    for j in jobs:
        assert type(j) is dict

        # Mandatory job attributes.
        for attr in ("primary_input", "primary_output", "summary_index"):
            array = take(j, attr)
            assert type(array) is list
            assert len(array) == 1
            assert type(array[0]) is str

        # Optional job attributes.
        for attr in ("additional_inputs", "additional_outputs", "imports"):
            array = take(j, attr)
            if array is KeyError:
                continue
            assert type(array) is list
            assert all(type(a) is str for a in array)

        assert len(j) == 0

    assert len(jdoc) == 0


if __name__ == "__main__":
    json_arg = Path(sys.argv[-1])
    distributor_args = sys.argv[1:-1]

    print(f"{distributor_args=}")

    # Load the DTLTO information from the input JSON file.
    with json_arg.open() as f:
        jdoc = json.load(f)

    # Write the input JSON to stdout.
    print(json.dumps(jdoc, indent=4))

    # Check the format of the JSON.
    validate(jdoc)
