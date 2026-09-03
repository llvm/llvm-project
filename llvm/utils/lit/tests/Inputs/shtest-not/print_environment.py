import os


def execute():
    for name in ["FOO", "BAR"]:
        print(name, "=", os.environ.get(name, "[undefined]"))
