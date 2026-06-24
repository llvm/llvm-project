import hashlib
import logging
import os
import subprocess
import re

from pathlib import Path
from unidiff import PatchSet


def compute_patch_hash(patch_path):
    with open(patch_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def should_rebuild(build_dir, patch_path, binary_path):
    new_hash = compute_patch_hash(patch_path)
    hash_file = os.path.join(build_dir, ".last_patch_hash")

    if not binary_path:
        return True

    binary_path = os.path.abspath(binary_path)

    if not os.path.exists(binary_path):
        return True

    if not os.path.exists(hash_file):
        return True

    with open(hash_file) as f:
        if f.read().strip() == new_hash:
            return False

    return True


def mark_build_success(build_dir, patch_path):
    with open(os.path.join(build_dir, ".last_patch_hash"), "w") as f:
        f.write(compute_patch_hash(patch_path))


def delete_profraw(inst_build_dir):
    abs_dir = os.path.abspath(inst_build_dir)
    try:
        subprocess.run(
            ["find", abs_dir, "-type", "f", "-name", "*.profraw", "-delete"], check=True
        )
        log("Older '.profraw' files successfully deleted.")
    except subprocess.CalledProcessError as e:
        log(f"Error: {e}")


def configure_logging(inst_build_dir):
    os.makedirs(inst_build_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(inst_build_dir, "patch_coverage.log"),
        level=logging.INFO,
        format="%(message)s",
    )


def log(*args):
    message = " ".join(map(str, args))
    logging.info(message)


def classify_tests(patch_path):
    patch_diff = Path(patch_path).read_text()

    unit_tests = re.findall(
        r"^\+\+\+ [ab]/(.*unittests.*\.(?:cpp|c))$",
        patch_diff,
        re.MULTILINE,
    )

    lit_tests = re.findall(
        r"^\+\+\+ [ab]/(.*test/.*\.(?:ll|mir|mlir|fir|test|txt|s|c|cpp|f90))$",
        patch_diff,
        re.MULTILINE,
    )

    return unit_tests, lit_tests


def target_name(patch_path, inst_build_dir):
    try:
        patch_diff = Path(patch_path).read_text()

        matches = re.findall(
            r"^\+\+\+ [ab]/(.*unittests.*\.(?:cpp|c))$",
            patch_diff,
            re.MULTILINE,
        )

        if not matches:
            return None

        # just take the first match
        suite_name = Path(matches[0]).parent.name
        return f"unittests/{suite_name}/{suite_name}Tests"

    except Exception as e:
        log(f"Error finding target name: {e}")
        sys.exit(1)


def get_projects_from_cache(cache_file):
    with open(cache_file) as f:
        for line in f:
            if line.startswith("LLVM_ENABLE_PROJECTS"):
                return line.split("=")[1].strip()
    return ""


def resolve_projects(projects, build_dir):
    if projects:
        return projects

    cache_file = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.exists(cache_file):
        return get_projects_from_cache(cache_file)

    return ""


def group_contiguous_lines(lines):
    groups = []
    current = [lines[0]]

    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1] + 1:
            current.append(lines[i])
        else:
            groups.append(current)
            current = [lines[i]]

    groups.append(current)
    return groups
