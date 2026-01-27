import json
import sys
import difflib
import getopt


class CommandLineArguments:
    def __init__(self):
        self.ignore_extra_keys = False
        self.context = 3
        self.expected_file = None
        self.actual_file = None


def fail(msg, exit_code):
    sys.stderr.write(msg)
    sys.stderr.write("\n")
    sys.exit(exit_code)


def readJson(filepath):
    def checkDuplicateKeys(pairs):
        keys = [key for (key, value) in pairs]
        seen = set()
        duplicates = set()

        for key in keys:
            if key in seen:
                duplicates.add(key)
            else:
                seen.add(key)

        if duplicates:
            dupkeys = ", ".join(sorted(duplicates))
            fail(f"Error: failed to read JSON. Found duplicate keys: {dupkeys}.", 2)

        return dict(pairs)

    try:
        with open(filepath, "r") as f:
            data = json.load(f, object_pairs_hook=checkDuplicateKeys)
            return data
    except Exception as e:
        fail(f"Error: failed to read JSON from {filepath}: {e}.", 2)


def normalizeJson(obj, indent=2):
    return json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False)


def pruneJson(expected, actual):
    if type(expected) != type(actual):
        return actual
    elif isinstance(expected, dict):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        result = {}
        for key in sorted(expected_keys & actual_keys):
            result[key] = pruneJson(expected[key], actual[key])
        return result
    elif isinstance(expected, list):
        result = []
        expected_len = len(expected)
        for i in range(len(actual)):
            if i < expected_len:
                result.append(pruneJson(expected[i], actual[i]))
            else:
                result.append(actual[i])
        return result
    else:
        return actual


def diffJson(expected, actual, expected_file, actual_file, context_lines):
    expected_lines = normalizeJson(expected).splitlines(keepends=True)
    actual_lines = normalizeJson(actual).splitlines(keepends=True)

    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile=expected_file,
        tofile=actual_file,
        n=context_lines,
        lineterm="",
    )

    green = "\x1b[0;32m"
    red = "\x1b[0;31m"
    normal = "\x1b[0m"

    colored_diff = []
    for line in diff:
        if line.startswith("+ "):
            colored_diff.append(f"{green}{line}{normal}")
        elif line.startswith("- "):
            colored_diff.append(f"{red}{line}{normal}")
        else:
            colored_diff.append(line)

    return "".join(colored_diff)


def compareJson(actual, expected, args):
    if args.ignore_extra_keys:
        actual = pruneJson(expected, actual)

    if actual != expected:
        diff = diffJson(
            expected, actual, args.expected_file, args.actual_file, args.context
        )
        fail(diff, 1)


def parseCommandLine():
    try:
        opts, args = getopt.gnu_getopt(
            sys.argv[1:], "ic:", ["ignore-extra-keys", "context="]
        )
    except getopt.GetoptError as err:
        fail(f"Error: failed to parse command-line arguments: {err}.", 2)

    flags = CommandLineArguments()

    for opt, arg in opts:
        if opt in ("i", "--ignore-extra-keys"):
            flags.ignore_extra_keys = True
        elif opt in ("-c", "--context"):
            try:
                flags.context = int(arg)
                if flags.context < 0:
                    raise ValueError()
            except ValueError:
                fail(f"Error: invalid context value: {arg}.", 2)

    if len(args) != 2:
        fail("Error: expected two positional arguments.", 2)

    flags.expected_file = args[0]
    flags.actual_file = args[1]

    return flags


def main():
    args = parseCommandLine()

    expected = readJson(args.expected_file)
    actual = readJson(args.actual_file)

    compareJson(actual, expected, args)


if __name__ == "__main__":
    main()
