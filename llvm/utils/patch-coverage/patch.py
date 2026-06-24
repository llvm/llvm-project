import subprocess
import sys
import re
import os

from utils import log
from unidiff import PatchSet


def create_patch_from_last_commits(output_path, num_commits):
    try:
        diff_cmd = ["git", "diff", f"HEAD~{num_commits}", "HEAD"]
        diff_output = subprocess.check_output(diff_cmd).decode("utf-8", "ignore")

        with open(output_path, "wb") as patch_file:
            patch_file.write(diff_output.encode("utf-8"))

        log(f"Patch file '{output_path}' created successfully.")
        log("")

    except subprocess.CalledProcessError as e:
        print("Error while creating the patch from the last commits:", e)
        sys.exit(1)


def extract_source_files_from_patch(patch_path):
    try:
        source_files = []
        test_files = []
        with open(patch_path, "rb") as patch_file:
            patch_diff = patch_file.read().decode("utf-8", "ignore")

        file_matches = re.findall(r"\+{3} b/(\S+)", patch_diff)
        known_source_file_extension = (".cpp", ".c", ".mm")

        for file in file_matches:
            if any(keyword in file.lower() for keyword in ["test", "unittest"]):
                test_files.append(file)
                continue
            if not file.lower().endswith(known_source_file_extension):
                continue
            else:
                repo_root = os.path.abspath(os.getcwd())
                full_path = os.path.join(repo_root, file)
                source_files.append(full_path)

        if not source_files:
            print("No source files found in the patch. Exiting.")
            sys.exit(1)

        if not test_files:
            print("No test files found in the patch. Exiting.")
            sys.exit(1)

        print("\n[patch-coverage] Source files in the patch:")
        for source_file in source_files:
            print(source_file)

        print("\n[patch-coverage] Test files in the patch:")
        for test_file in test_files:
            print(test_file)
        print("\n")

        return source_files

    except Exception as ex:
        print("Error while extracting files from patch:", ex)
        sys.exit(1)


def extract_modified_source_lines_from_patch(patch_path, tests):
    source_lines = {}

    tests_set = {os.path.abspath(os.path.normpath(t)) for t in tests}
    repo_root = os.path.abspath(os.getcwd())

    try:
        patchset = PatchSet.from_filename(patch_path)

        for patched_file in patchset:
            target_path = patched_file.path

            current_file = os.path.abspath(
                os.path.normpath(os.path.join(repo_root, target_path))
            )

            if current_file in tests_set or patched_file.is_removed_file:
                continue

            lines = []

            for hunk in patched_file:
                for line in hunk:
                    if line.is_added:
                        lines.append((line.target_line_no, line.value.rstrip("\n")))

            if lines:
                source_lines[current_file] = lines

        return source_lines

    except Exception as ex:
        print(f"[patch-coverage] Failed to parse patch {patch_path}: {ex}")
        return {}


def write_source_file_allowlist(source_files, allowlist_path):
    try:
        lines = []
        for source_file in source_files:
            lines.append(f"source:{source_file}=allow\n")

        lines.append("default:skip\n")

        new_content = "".join(lines)

        if os.path.exists(allowlist_path):
            with open(allowlist_path) as f:
                if f.read() == new_content:
                    log("Allowlist unchanged, skipping write.")
                    log("")
                    return

        with open(allowlist_path, "w") as f:
            f.write(new_content)

        log(f"Source file allowlist written to '{allowlist_path}'.\n")

    except Exception as e:
        log(f"Error while writing allow list: {e}")
        sys.exit(1)
