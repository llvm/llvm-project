# RUN: %{lit} --update-tests --ignore-fail -v %S/Inputs/pass-test-update | FileCheck %s --implicit-check-not Exception

# CHECK:      UNRESOLVED: pass-test-update :: fail.test (1 of 5)
# CHECK-NEXT: ******************** TEST 'pass-test-update :: fail.test' FAILED ********************
# CHECK-NEXT: # {{R}}UN: at line 1
# CHECK-NEXT: not echo "fail"
# CHECK-NEXT: # executed command: not echo fail
# CHECK-NEXT: # .---command stdout------------
# CHECK-NEXT: # | fail
# CHECK-NEXT: # `-----------------------------
# CHECK-NEXT: # error: command failed with exit status: 1
# CHECK-NEXT: Exception occurred in test updater:
# CHECK-NEXT: Traceback (most recent call last):
# CHECK-NEXT:   File {{.*}}, line {{.*}}, in {{.*}}
# CHECK-NEXT:     update_output = test_updater(result, test)
# CHECK-NEXT:   File "{{.*}}/should_not_run.py", line {{.*}}, in should_not_run
# CHECK-NEXT:     raise Exception("this test updater should only run on failure")
# CHECK-NEXT: Exception: this test updater should only run on failure
# CHECK-EMPTY:
# CHECK-NEXT: ********************
# CHECK-NEXT: PASS: pass-test-update :: pass-silent.test (2 of 5)
# CHECK-NEXT: PASS: pass-test-update :: pass.test (3 of 5)
# CHECK-NEXT: {{X}}FAIL: pass-test-update :: xfail.test (4 of 5)
# CHECK-NEXT: XPASS: pass-test-update :: xpass.test (5 of 5)
# CHECK-NEXT: ******************** TEST 'pass-test-update :: xpass.test' FAILED ********************
# CHECK-NEXT: Exit Code: 0
# CHECK-EMPTY:
# CHECK-NEXT: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: # {{R}}UN: at line 2
# CHECK-NEXT: echo "accidentally passed"
# CHECK-NEXT: # executed command: echo 'accidentally passed'
# CHECK-NEXT: # .---command stdout------------
# CHECK-NEXT: # | accidentally passed
# CHECK-NEXT: # `-----------------------------
# CHECK-EMPTY:
# CHECK-NEXT: --
# CHECK-EMPTY:
# CHECK-NEXT: ********************
