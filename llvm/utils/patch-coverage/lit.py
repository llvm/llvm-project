import subprocess
import os


def parse_suite_info(s):
    curr_suite = None
    res = {}

    for line in s.decode().splitlines():
        leading_spaces = len(line) - len(line.lstrip(" "))

        if leading_spaces == 2 and line.split():
            curr_suite = line.split()[0]
        elif curr_suite is not None and leading_spaces == 4 and "Source Root:" in line:
            if curr_suite in res:
                raise RuntimeError(f"Duplicate suite detected: {curr_suite}")
            res[curr_suite] = line.split()[-1]

    return res


def find_lit_tests(lit_path, test_paths):
    suites_cmd = [lit_path, "--show-suites"] + test_paths
    output = subprocess.check_output(suites_cmd)

    # `--show-suites` produce output in the following form  -
    #  LLVM - 61914 tests
    #    Source Root: /Users/<username>/llvm-project/llvm/test
    #
    # Parse it to construct following format -
    # {'LLVM': '/Users/<username>/llvm-project/llvm/test'}
    test_suites = parse_suite_info(output)

    tests_cmd = [lit_path, "--show-tests"] + test_paths
    output = subprocess.check_output(tests_cmd)

    lines = [line.decode() for line in output.splitlines()]

    test_info = [line.split() for line in lines if "::" in line]

    # Construct absolute path of each test case of test suite.
    if test_info is not None and len(test_info) > 0:
        if len(test_info[0]) == 3:
            return [
                os.path.join(test_suites[suite], test_case)
                for (suite, sep, test_case) in test_info
            ]
        elif len(test_info[0]) == 4:
            return [
                os.path.join(test_suites[suite1], test_case)
                for (suite1, suite2, sep, test_case) in test_info
            ]
    else:
        return []
