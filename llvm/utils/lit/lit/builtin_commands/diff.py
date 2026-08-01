import difflib
import functools
import getopt
import locale
import os
import re
import sys

# diff.py runs in two modes during the in-process migration:
#   - In-process: imported as 'lit.builtin_commands.diff', so __package__ is
#     set and the relative import resolves to lit.util from the source tree.
#   - Spawned fallback: run as __main__ with __package__=None, so the relative
#     import raises ImportError and the fallback picks up util.py via PYTHONPATH
#     pointing at the lit/ directory.
# A relative import is used (not 'import lit.util') to avoid accidentally
# importing a system-installed lit package that may lack abs_path_preserve_drive.
# TODO: Collapse to 'from .. import util' once standalone spawning is removed.

try:
    from .. import util
except ImportError:
    import util


class DiffFlags:
    # TODO(prasoon054): Replace __slots__ with @dataclass(slots=True)
    # once the minimum Python version is bumped to 3.10.
    # https://github.com/llvm/llvm-project/issues/200531
    __slots__ = (
        "ignore_all_space",
        "ignore_space_change",
        "ignore_matching_lines",
        "ignore_matching_lines_regex",
        "unified_diff",
        "num_context_lines",
        "recursive_diff",
        "strip_trailing_cr",
    )

    def __init__(self):
        self.ignore_all_space = False
        self.ignore_space_change = False
        self.ignore_matching_lines = False
        self.ignore_matching_lines_regex = ""
        self.unified_diff = False
        self.num_context_lines = 3
        self.recursive_diff = False
        self.strip_trailing_cr = False


def getDirTree(path, basedir=""):
    # Tree is a tuple of form (dirname, child_trees).
    # An empty dir has child_trees = [], a file has child_trees = None.
    child_trees = []
    for dirname, child_dirs, files in os.walk(os.path.join(basedir, path)):
        for child_dir in child_dirs:
            child_trees.append(getDirTree(child_dir, dirname))
        for filename in files:
            child_trees.append((filename, None))
        return path, sorted(child_trees)


def compareTwoFiles(flags, filepaths, stdin, stdout):
    filelines = []
    for file in filepaths:
        if file == "-":
            filelines.append(stdin.readlines())
        else:
            with open(file, "rb") as file_bin:
                filelines.append(file_bin.readlines())

    try:
        return compareTwoTextFiles(
            flags, filepaths, filelines, locale.getpreferredencoding(False), stdout
        )
    except UnicodeDecodeError:
        try:
            return compareTwoTextFiles(flags, filepaths, filelines, "utf-8", stdout)
        except:
            return compareTwoBinaryFiles(flags, filepaths, filelines, stdout)


def compareTwoBinaryFiles(flags, filepaths, filelines, stdout):
    exitCode = 0
    diffs = difflib.diff_bytes(
        difflib.unified_diff,
        filelines[0],
        filelines[1],
        filepaths[0].encode(),
        filepaths[1].encode(),
        n=flags.num_context_lines,
    )

    for diff in diffs:
        stdout.write(
            diff.decode(errors="backslashreplace").encode(
                locale.getpreferredencoding(False)
            )
        )
        exitCode = 1
    return exitCode


def compareTwoTextFiles(flags, filepaths, filelines_bin, encoding, stdout):
    filelines = []
    for lines_bin in filelines_bin:
        lines = []
        for line_bin in lines_bin:
            line = line_bin.decode(encoding=encoding)
            lines.append(line)
        filelines.append(lines)

    exitCode = 0

    def compose2(f, g):
        return lambda x: f(g(x))

    f = lambda x: x
    if flags.strip_trailing_cr:
        f = compose2(lambda line: line.replace("\r\n", "\n"), f)
    if flags.ignore_all_space or flags.ignore_space_change:
        ignoreSpace = lambda line, separator: separator.join(line.split()) + "\n"
        ignoreAllSpaceOrSpaceChange = functools.partial(
            ignoreSpace, separator="" if flags.ignore_all_space else " "
        )
        f = compose2(ignoreAllSpaceOrSpaceChange, f)

    for idx, lines in enumerate(filelines):
        if flags.ignore_matching_lines:
            lines = filter(
                lambda x: not re.match(
                    r"{}".format(flags.ignore_matching_lines_regex), x
                ),
                lines,
            )
        filelines[idx] = [f(line) for line in lines]

    func = difflib.unified_diff if flags.unified_diff else difflib.context_diff
    for diff in func(
        filelines[0],
        filelines[1],
        filepaths[0],
        filepaths[1],
        n=flags.num_context_lines,
    ):
        stdout.write(diff.encode(encoding))
        exitCode = 1
    return exitCode


def printDirVsFile(dir_path, file_path, stdout):
    if os.path.getsize(file_path):
        msg = "File %s is a directory while file %s is a regular file"
    else:
        msg = "File %s is a directory while file %s is a regular empty file"
    stdout.write(
        (msg % (dir_path, file_path) + "\n").encode(locale.getpreferredencoding(False))
    )


def printFileVsDir(file_path, dir_path, stdout):
    if os.path.getsize(file_path):
        msg = "File %s is a regular file while file %s is a directory"
    else:
        msg = "File %s is a regular empty file while file %s is a directory"
    stdout.write(
        (msg % (file_path, dir_path) + "\n").encode(locale.getpreferredencoding(False))
    )


def printOnlyIn(basedir, path, name, stdout):
    stdout.write(
        ("Only in %s: %s\n" % (os.path.join(basedir, path), name)).encode(
            locale.getpreferredencoding(False)
        )
    )


def compareDirTrees(flags, dir_trees, stdin, stdout, base_paths=["", ""]):
    # Dirnames of the trees are not checked, it's caller's responsibility,
    # as top-level dirnames are always different. Base paths are important
    # for doing os.walk, but we don't put it into tree's dirname in order
    # to speed up string comparison below and while sorting in getDirTree.
    left_tree, right_tree = dir_trees[0], dir_trees[1]
    left_base, right_base = base_paths[0], base_paths[1]

    # Compare two files or report file vs. directory mismatch.
    if left_tree[1] is None and right_tree[1] is None:
        return compareTwoFiles(
            flags,
            [
                os.path.join(left_base, left_tree[0]),
                os.path.join(right_base, right_tree[0]),
            ],
            stdin,
            stdout,
        )

    if left_tree[1] is None and right_tree[1] is not None:
        printFileVsDir(
            os.path.join(left_base, left_tree[0]),
            os.path.join(right_base, right_tree[0]),
            stdout,
        )
        return 1

    if left_tree[1] is not None and right_tree[1] is None:
        printDirVsFile(
            os.path.join(left_base, left_tree[0]),
            os.path.join(right_base, right_tree[0]),
            stdout,
        )
        return 1

    # Compare two directories via recursive use of compareDirTrees.
    exitCode = 0
    left_names = [node[0] for node in left_tree[1]]
    right_names = [node[0] for node in right_tree[1]]
    l, r = 0, 0
    while l < len(left_names) and r < len(right_names):
        # Names are sorted in getDirTree, rely on that order.
        if left_names[l] < right_names[r]:
            exitCode = 1
            printOnlyIn(left_base, left_tree[0], left_names[l], stdout)
            l += 1
        elif left_names[l] > right_names[r]:
            exitCode = 1
            printOnlyIn(right_base, right_tree[0], right_names[r], stdout)
            r += 1
        else:
            exitCode |= compareDirTrees(
                flags,
                [left_tree[1][l], right_tree[1][r]],
                stdin,
                stdout,
                [
                    os.path.join(left_base, left_tree[0]),
                    os.path.join(right_base, right_tree[0]),
                ],
            )
            l += 1
            r += 1

    # At least one of the trees has ended. Report names from the other tree.
    while l < len(left_names):
        exitCode = 1
        printOnlyIn(left_base, left_tree[0], left_names[l], stdout)
        l += 1
    while r < len(right_names):
        exitCode = 1
        printOnlyIn(right_base, right_tree[0], right_names[r], stdout)
        r += 1
    return exitCode


def run(argv, stdin, stdout, stderr, cwd):
    """In-process diff.

    Writes bytes to stdout, returns an exit code, and never calls sys.exit.
    This makes it safe to run inside lit worker as well as from the
    standalone main below.

    Args:
        argv: Command-line arguments. The first element is the command name,
            followed by options and the two files or directories to compare.
        stdin: Binary input stream, used if a file operand is '-'.
        stdout: Binary output stream for the diff output.
        stderr: Binary error stream for error messages.
        cwd: The shell's current working directory, used to resolve relative
            file paths.

    Returns:
        An integer representing the exit code (0 for no differences, 1 if
        differences were found or an error occurred).
    """
    args = argv[1:]
    try:
        opts, args = getopt.gnu_getopt(args, "wbuI:U:r", ["strip-trailing-cr"])
    except getopt.GetoptError as err:
        stderr.write(b"Unsupported: 'diff': %s\n" % str(err).encode())
        return 1

    flags = DiffFlags()
    filepaths, dir_trees = [], []
    for o, a in opts:
        if o == "-w":
            flags.ignore_all_space = True
        elif o == "-b":
            flags.ignore_space_change = True
        elif o == "-u":
            flags.unified_diff = True
        elif o.startswith("-U"):
            flags.unified_diff = True
            try:
                flags.num_context_lines = int(a)
                if flags.num_context_lines < 0:
                    raise ValueError
            except:
                stderr.write(b"Error: invalid '-U' argument: %s\n" % a.encode())
                return 1
        elif o == "-I":
            flags.ignore_matching_lines = True
            flags.ignore_matching_lines_regex = a
        elif o == "-r":
            flags.recursive_diff = True
        elif o == "--strip-trailing-cr":
            flags.strip_trailing_cr = True
        else:
            assert False, "unhandled option"

    if len(args) != 2:
        stderr.write(b"Error: missing or extra operand\n")
        return 1

    exitCode = 0
    try:
        for file in args:
            if file != "-" and not os.path.isabs(file):
                file = util.abs_path_preserve_drive(os.path.join(cwd, file))

            if flags.recursive_diff:
                if file == "-":
                    stderr.write(b"Error: cannot recursively compare '-'\n")
                    return 1
                dir_trees.append(getDirTree(file))
            else:
                filepaths.append(file)

        if not flags.recursive_diff:
            exitCode = compareTwoFiles(flags, filepaths, stdin, stdout)
        else:
            exitCode = compareDirTrees(flags, dir_trees, stdin, stdout)

    except IOError as err:
        stderr.write(b"Error: 'diff' command failed, %s\n" % str(err).encode())
        exitCode = 1

    return exitCode


def main(argv):
    out = getattr(sys.stdout, "buffer", sys.stdout)
    sys.exit(run(argv, sys.stdin.buffer, out, sys.stderr.buffer, os.getcwd()))

if __name__ == "__main__":
    main(sys.argv)
