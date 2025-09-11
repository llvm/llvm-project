import shutil
import os
import shlex
import pathlib

"""
This file provides the `diff_test_updater` function, which is invoked on failed RUN lines when lit is executed with --update-tests.
It checks whether the failed command is `diff` and, if so, uses heuristics to determine which file is the checked-in reference file and which file is output from the test case.
The heuristics are currently as follows:
    - if exactly one file originates from the `split-file` command, that file is the reference file and the other is the output file
    - if exactly one file ends with ".expected" (common pattern in LLVM), that file is the reference file and the other is the output file
    - if exactly one file path contains ".tmp" (e.g. because it contains the expansion of "%t"), that file is the reference file and the other is the output file
If the command matches one of these patterns the output file content is copied to the reference file to make the test pass.
If the reference file originated in `split-file`, the output file content is instead copied to the corresponding slice of the test file.
Otherwise the test is ignored.

Possible improvements:
    - Support stdin patterns like "my_binary %s | diff expected.txt"
    - Scan RUN lines to see if a file is the source of output from a previous command (other than `split-file`).
      If it is then it is not a reference file that can be copied to, regardless of name, since the test will overwrite it anyways.
    - Only update the parts that need updating (based on the diff output). Could help avoid noisy updates when e.g. whitespace changes are ignored.
"""


class NormalFileTarget:
    def __init__(self, target):
        self.target = target

    def copyFrom(self, source):
        shutil.copy(source, self.target)

    def __str__(self):
        return self.target


class SplitFileTarget:
    def __init__(self, slice_start_idx, test_path, lines):
        self.slice_start_idx = slice_start_idx
        self.test_path = test_path
        self.lines = lines

    def copyFrom(self, source):
        lines_before = self.lines[: self.slice_start_idx + 1]
        self.lines = self.lines[self.slice_start_idx + 1 :]
        slice_end_idx = None
        for i, l in enumerate(self.lines):
            if SplitFileTarget._get_split_line_path(l) != None:
                slice_end_idx = i
                break
        if slice_end_idx is not None:
            lines_after = self.lines[slice_end_idx:]
        else:
            lines_after = []
        with open(source, "r") as f:
            new_lines = lines_before + f.readlines() + lines_after
        with open(self.test_path, "w") as f:
            for l in new_lines:
                f.write(l)

    def __str__(self):
        return f"slice in {self.test_path}"

    @staticmethod
    def get_target_dir(commands, test_path):
        for cmd in commands:
            split = shlex.split(cmd)
            if "split-file" not in split:
                continue
            start_idx = split.index("split-file")
            split = split[start_idx:]
            if len(split) < 3:
                continue
            if split[1].strip() != test_path:
                continue
            return split[2].strip()
        return None

    @staticmethod
    def create(path, commands, test_path, target_dir):
        path = pathlib.Path(path)
        with open(test_path, "r") as f:
            lines = f.readlines()
        for i, l in enumerate(lines):
            p = SplitFileTarget._get_split_line_path(l)
            if p and path.samefile(os.path.join(target_dir, p)):
                idx = i
                break
        else:
            return None
        return SplitFileTarget(idx, test_path, lines)

    @staticmethod
    def _get_split_line_path(l):
        if len(l) < 6:
            return None
        if l.startswith("//"):
            l = l[2:]
        else:
            l = l[1:]
        if l.startswith("--- "):
            l = l[4:]
        else:
            return None
        return l.rstrip()


def get_source_and_target(a, b, test_path, commands):
    """
    Try to figure out which file is the test output and which is the reference.
    """
    split_target_dir = SplitFileTarget.get_target_dir(commands, test_path)
    if split_target_dir:
        a_target = SplitFileTarget.create(a, commands, test_path, split_target_dir)
        b_target = SplitFileTarget.create(b, commands, test_path, split_target_dir)
        if a_target and b_target:
            return None
        if a_target:
            return b, a_target
        if b_target:
            return a, b_target

    expected_suffix = ".expected"
    if a.endswith(expected_suffix) and not b.endswith(expected_suffix):
        return b, NormalFileTarget(a)
    if b.endswith(expected_suffix) and not a.endswith(expected_suffix):
        return a, NormalFileTarget(b)

    tmp_substr = ".tmp"
    if tmp_substr in a and not tmp_substr in b:
        return a, NormalFileTarget(b)
    if tmp_substr in b and not tmp_substr in a:
        return b, NormalFileTarget(a)

    return None


def filter_flags(args):
    return [arg for arg in args if not arg.startswith("-")]


def diff_test_updater(result, test, commands):
    args = filter_flags(result.command.args)
    if len(args) != 3:
        return None
    [cmd, a, b] = args
    if cmd != "diff":
        return None
    res = get_source_and_target(a, b, test.getFilePath(), commands)
    if not res:
        return f"update-diff-test: could not deduce source and target from {a} and {b}"
    source, target = res
    target.copyFrom(source)
    return f"update-diff-test: copied {source} to {target}"
