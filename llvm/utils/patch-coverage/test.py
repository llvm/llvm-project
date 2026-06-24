import re
import os
import subprocess
import sys

from pathlib import Path
from utils import log
from utils import target_name

# TODO: We should run llvm-lit in batch mode.


def run_single_test_with_coverage(llvm_lit_path, test_path, inst_build_dir):
    try:
        profile_dir = os.path.abspath(os.path.join(inst_build_dir, "profiles"))
        os.makedirs(profile_dir, exist_ok=True)

        test_env = os.environ.copy()
        test_env["LLVM_PROFILE_FILE"] = os.path.join(profile_dir, "test-%p.profraw")

        lit_cmd = [llvm_lit_path, "-v", test_path]
        subprocess.check_call(lit_cmd, env=test_env)

        log("Test case executed:", test_path)

    except subprocess.CalledProcessError as e:
        log("Error while running test:", e)
        sys.exit(1)

    except Exception as ex:
        log("Error:", ex)
        sys.exit(1)


def run_modified_lit_tests(llvm_lit_path, patch_path, tests, inst_build_dir):
    try:
        with open(patch_path, "r") as patch_file:
            patch_diff = patch_file.read()

        tests_set = {os.path.normpath(t) for t in tests}

        modified_tests = set()

        for match in re.finditer(
            r"^\+\+\+\s+(?:[ab]/)?(.+)$",
            patch_diff,
            re.MULTILINE,
        ):
            file_path = match.group(1).strip()

            # Skip deleted files
            if file_path == "/dev/null":
                continue

            if not file_path.endswith(
                (
                    ".ll",
                    ".mir",
                    ".mlir",
                    ".fir",
                    ".test",
                    ".s",
                    ".c",
                    ".cpp",
                    ".f90",
                    ".py",
                )
            ):
                continue

            abs_path = os.path.normpath(
                os.path.abspath(os.path.join(os.getcwd(), file_path))
            )
            if abs_path in tests_set:
                modified_tests.add(abs_path)
                log("Lit test file in patch:", abs_path)

        if not modified_tests:
            log("No modified lit tests found in the patch.")
            return

        log("\nRunning modified test cases:")

        failures = []
        for test_file in sorted(modified_tests):
            try:
                run_single_test_with_coverage(llvm_lit_path, test_file, inst_build_dir)
            except SystemExit:
                failures.append(test_file)

        if failures:
            log("\nFailed tests:")
            for f in failures:
                log(" -", f)
            sys.exit(1)

    except Exception as ex:
        log("Error:", ex)
        sys.exit(1)


def run_modified_unit_tests(build_dir, inst_build_dir, patch_path):
    log("Starting unit test execution...")

    try:
        target = target_name(patch_path, inst_build_dir)

        if not target:
            log("No unit test target found.")
            return

        binary_path = os.path.join(inst_build_dir, target)
        test_name = Path(target).name
        profraw_path = Path(inst_build_dir) / f"{test_name}.profraw"

        subprocess.check_call(["ninja", "-C", inst_build_dir, target])

        env = os.environ.copy()
        env["LLVM_PROFILE_FILE"] = str(profraw_path)

        log(f"Executing modified unit test case target: {binary_path}")
        subprocess.check_call([binary_path], env=env)

    except subprocess.CalledProcessError as e:
        log(f"Error while running unit test {target}: {e}")
        sys.exit(1)

    except Exception as ex:
        log("Error:", ex)
        sys.exit(1)
