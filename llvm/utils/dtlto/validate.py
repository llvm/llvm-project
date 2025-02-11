import sys
import json
from pathlib import Path


def take(jvalue, jpath):
    parts = jpath.split(".")
    for part in parts[:-1]:
        jvalue = jvalue[part]
    return jvalue.pop(parts[-1], KeyError)


if __name__ == "__main__":
    json_arg = sys.argv[-1]
    distributor_args = sys.argv[1:-1]

    print(f"{distributor_args=}")

    # Load the DTLTO information from the input JSON file.
    jdoc = json.loads(Path(json_arg).read_bytes())

    # Write the input JSON to stdout.
    print(json.dumps(jdoc, indent=4))

    # Check the format of the JSON
    assert type(take(jdoc, "common.linker_output")) is str
    assert type(take(jdoc, "common.linker_version")) is str

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
