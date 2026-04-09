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
    cwd = os.getcwd()
    os.chdir(inst_build_dir)

    command = 'find . -type f -name "*.profraw" -delete'
    os.chdir(cwd)

    try:
        subprocess.run(command, shell=True, check=True)
        log("Older '.profraw' files are successfully deleted.")
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
        r"^\+\+\+ [ab]/(llvm/test/.*\.(?:ll|mir|mlir|fir|test|txt))$",
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
