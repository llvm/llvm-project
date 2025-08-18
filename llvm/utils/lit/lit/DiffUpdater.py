import shutil

"""
This file provides the `diff_test_updater` function, which is invoked on failed RUN lines when lit is executed with --update-tests.
It checks whether the failed command is `diff` and, if so, uses heuristics to determine which file is the checked-in reference file and which file is output from the test case.
The heuristics are currently as follows:
    - if exactly one file ends with ".expected" (common pattern in LLVM), that file is the reference file and the other is the output file
    - if exactly one file path contains ".tmp" (e.g. because it contains the expansion of "%t"), that file is the reference file and the other is the output file
If the command matches one of these patterns the output file content is copied to the reference file to make the test pass.
Otherwise the test is ignored.

Possible improvements:
    - Support stdin patterns like "my_binary %s | diff expected.txt"
    - Scan RUN lines to see if a file is the source of output from a previous command.
      If it is then it is not a reference file that can be copied to, regardless of name, since the test will overwrite it anyways.
    - Only update the parts that need updating (based on the diff output). Could help avoid noisy updates when e.g. whitespace changes are ignored.
"""


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
