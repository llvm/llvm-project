import json
import sys


def to_json(fname: str):
    with open(fname) as f:
        return json.load(f)


a = to_json(sys.argv[1])
b = to_json(sys.argv[2])

if a == b:
    exit(0)
exit(1)
