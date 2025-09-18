# RUN: %{lit} --update-tests --ignore-fail -v %S/Inputs/pass-test-update | FileCheck %s --implicit-check-not Exception

# CHECK: UNRESOLVED: pass-test-update :: fail.test (1 of 5)
# CHECK: ******************** TEST 'pass-test-update :: fail.test' FAILED ********************
# CHECK: # {{R}}UN: at line 1
# CHECK: not echo "fail"
# CHECK: # executed command: not echo fail
# CHECK: # .---command stdout------------
# CHECK: # | fail
# CHECK: # `-----------------------------
# CHECK: # error: command failed with exit status: 1
# CHECK: Exception occurred in test updater:
# CHECK: Traceback (most recent call last):
# CHECK:   File {{.*}}, line {{.*}}, in {{.*}}
# CHECK:     update_output = test_updater(result, test, commands)
# CHECK:   File "{{.*}}{{/|\\}}should_not_run.py", line {{.*}}, in should_not_run
# CHECK:     raise Exception("this test updater should only run on failure")
# CHECK: Exception: this test updater should only run on failure
# CHECK: ********************
# CHECK: PASS: pass-test-update :: pass-silent.test (2 of 5)
# CHECK: PASS: pass-test-update :: pass.test (3 of 5)
# CHECK: {{X}}FAIL: pass-test-update :: xfail.test (4 of 5)
# CHECK: XPASS: pass-test-update :: xpass.test (5 of 5)
# CHECK: ******************** TEST 'pass-test-update :: xpass.test' FAILED ********************
# CHECK: Exit Code: 0
# CHECK: Command Output (stdout):
# CHECK: --
# CHECK: # {{R}}UN: at line 2
# CHECK: echo "accidentally passed"
# CHECK: # executed command: echo 'accidentally passed'
# CHECK: # .---command stdout------------
# CHECK: # | accidentally passed
# CHECK: # `-----------------------------
# CHECK: ********************
