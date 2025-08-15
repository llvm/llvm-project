import shutil


def get_source_and_target(a, b):
    """
    Try to figure out which file is the test output and which is the reference.
    """
    expected_suffix = ".expected"
    if a.endswith(expected_suffix) and not b.endswith(expected_suffix):
        return b, a
    if b.endswith(expected_suffix) and not a.endswith(expected_suffix):
        return a, b

    tmp_substr = ".tmp"
    if tmp_substr in a and not tmp_substr in b:
        return a, b
    if tmp_substr in b and not tmp_substr in a:
        return b, a

    return None


def filter_flags(args):
    return [arg for arg in args if not arg.startswith("-")]


def diff_test_updater(result, test):
    args = filter_flags(result.command.args)
    if len(args) != 3:
        return None
    [cmd, a, b] = args
    if cmd != "diff":
        return None
    res = get_source_and_target(a, b)
    if not res:
        return f"update-diff-test: could not deduce source and target from {a} and {b}"
    source, target = res
    shutil.copy(source, target)
    return f"update-diff-test: copied {source} to {target}"
