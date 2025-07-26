import argparse
import os
import re
import subprocess
import sys

def is_text(file):
    text_file_extensions = {
        ".apinotes",
        ".asm",
        ".bazel",
        ".c",
        ".cc",
        ".cfg",
        ".cl",
        ".clcpp",
        ".cmake",
        ".cmd",
        ".cpp",
        ".cppm",
        ".css",
        ".csv",
        ".cu",
        ".d",
        ".def",
        ".dot",
        ".expected",
        ".f",
        ".f90",
        ".fir",
        ".gn",
        ".gni",
        ".h",
        ".hip",
        ".hlsl",
        ".hpp",
        ".html",
        ".i",
        ".in",
        ".inc",
        ".jscop",
        ".json",
        ".ll",
        ".m",
        ".map",
        ".md",
        ".mir",
        ".mlir",
        ".mm",
        ".modulemap",
        ".plist",
        ".py",
        ".rc",
        ".result",
        ".rsp",
        ".rst",
        ".s",
        ".script",
        ".sh",
        ".st",
        ".tbd",
        ".td",
        ".template",
        ".test",
        ".transformed",
        ".txt",
        ".xml",
        ".yml",
        ".yaml",
    }
    _, ext = os.path.splitext(file)
    return ext.lower() in text_file_extensions

def check_file(path, fix):
    try:
        trailing = False
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line_number, line in enumerate(f, 1):
                if line.rstrip("\n").endswith(" "):
                    print(f"{path}:{line_number}: Trailing whitespace found")
                    trailing = True
        if trailing and fix:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.rstrip(" \t") for line in f]
            with open(path, "w", encoding="utf-8", errors="ignore") as f:
                f.writelines(lines)
        return trailing
    except UnicodeDecodeError:
        print(f"Warning: Encoding error encountered for {path}")
    except FileNotFoundError:
        print(f"Warning: Could not open {path}")
    return False

def check_paths(paths, exclude, fix):
    exclude = [os.path.abspath(d) for d in exclude]
    seen = set()
    found_trailing = False
    for path in paths:
        if os.path.abspath(path) in exclude:
            continue
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) not in exclude]
            for file in files:
                file_path = os.path.join(root, file)
                if not is_text(file):
                    continue
                if file_path in seen:
                    continue
                seen.add(file_path)
                if check_file(file_path, fix):
                    found_trailing = True
    return found_trailing

HUNK_HEADER_REGEX = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DIFF_HEADER_REGEX = re.compile(r"^diff --git a/(.+) b/(.+)")

def parse_diffs(diff_text):
    diffs: dict[str, list[tuple[int, str]]] = dict()
    file = None
    current_line = None
    for line in diff_text.splitlines():
        match = DIFF_HEADER_REGEX.match(line)
        if match:
            file = match.groups()[1]
            if file not in diffs:
                diffs[file] = []
            current_line = None
            continue
        match = HUNK_HEADER_REGEX.match(line)
        if match:
            current_line = int(match.groups()[2])
            continue
        if not current_line: # haven't seen the hunk header yet, continue
            continue
        if line.startswith("+"):
            line = line[1:]
            diffs[file].append((current_line, line))
            current_line += 1
    return diffs

def check_paths_diff(paths, exclude, rev_start, rev_end):
    exclude = [os.path.abspath(d) for d in exclude]
    cmd = ["git", "diff", "-U0", rev_start, rev_end, *paths]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, encoding="utf-8")
    all_diffs = parse_diffs(proc.stdout)
    found_trailing = False
    for file, diffs in all_diffs.items():
        if any([os.path.abspath(file).startswith(path) for path in exclude]):
            continue
        # kind of redundant, diffs are text, but just to be consistent
        if not is_text(file):
            continue
        for num, line in diffs:
            if line.endswith(" "):
                print(f"{file}:{num}: Trailing whitespace found")
                found_trailing = True
    return found_trailing

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix", action=argparse.BooleanOptionalAction, default=False, help="Automatically apply fixes"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Paths to exclude. Can be used multiple times."
    )
    parser.add_argument(
        "--diff",
        help="Compute only based on changed lines (format: rev_start..rev_end)",
    )
    parser.add_argument("paths", nargs="*", type=str, help="Paths to check")

    args = parser.parse_args()
    paths = set(args.paths)

    if len(paths) == 0:
        print("Error: Must specify paths to check", file=sys.stderr)
        sys.exit(1)

    if args.diff and args.fix:
        print("Error: Diff mode doesn't support --fix currently", file=sys.stderr)
        sys.exit(1)

    if args.diff:
        rev_start, rev_end = args.diff.split("..")
        found_trailing = check_paths_diff(paths, args.exclude, rev_start, rev_end)
    else:
        found_trailing = check_paths(paths, args.exclude, args.fix)

    if found_trailing:
        sys.exit(1)
