# UNSUPPORTED: windows
# Check the various features of the ShTest format.
#
# RUN: not %{lit} -j 1 -v %{inputs}/shtest-format > %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK: -- Testing:

# CHECK: PASS: shtest-format :: argv0.txt
# CHECK: FAIL: shtest-format :: external_shell/fail.txt
# CHECK-NEXT: *** TEST 'shtest-format :: external_shell/fail.txt' FAILED ***
# CHECK: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: line 1: failed test output on stdout
# CHECK-NEXT: line 2: failed test output on stdout
# CHECK: Command Output (stderr):
# CHECK-NEXT: --
# CHECK-NEXT: cat: does-not-exist: No such file or directory
# CHECK: --

# CHECK: FAIL: shtest-format :: external_shell/fail_with_bad_encoding.txt
# CHECK-NEXT: *** TEST 'shtest-format :: external_shell/fail_with_bad_encoding.txt' FAILED ***
# CHECK: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: a line with bad encoding:
# CHECK: --

# CHECK: PASS: shtest-format :: external_shell/pass.txt

# CHECK: FAIL: shtest-format :: fail.txt
# CHECK-NEXT: *** TEST 'shtest-format :: fail.txt' FAILED ***
# CHECK-NEXT: Script:
# CHECK-NEXT: --
# CHECK-NEXT: printf "line 1
# CHECK-NEXT: false
# CHECK-NEXT: --
# CHECK-NEXT: Exit Code: 1
#
# CHECK: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: $ "printf"
# CHECK-NEXT: # command output:
# CHECK-NEXT: line 1: failed test output on stdout
# CHECK-NEXT: line 2: failed test output on stdout

# CHECK: UNRESOLVED: shtest-format :: no-test-line.txt
# CHECK: PASS: shtest-format :: pass.txt
# CHECK: UNSUPPORTED: shtest-format :: requires-any-missing.txt
# CHECK: PASS: shtest-format :: requires-any-present.txt
# CHECK: UNSUPPORTED: shtest-format :: requires-missing.txt
# CHECK: PASS: shtest-format :: requires-present.txt
# CHECK: UNRESOLVED: shtest-format :: requires-star.txt
# CHECK: UNSUPPORTED: shtest-format :: requires-triple.txt
# CHECK: PASS: shtest-format :: unsupported-expr-false.txt
# CHECK: UNSUPPORTED: shtest-format :: unsupported-expr-true.txt
# CHECK: UNRESOLVED: shtest-format :: unsupported-star.txt
# CHECK: UNSUPPORTED: shtest-format :: unsupported_dir/some-test.txt
# CHECK: PASS: shtest-format :: xfail-expr-false.txt
# CHECK: XFAIL: shtest-format :: xfail-expr-true.txt
# CHECK: XFAIL: shtest-format :: xfail-feature.txt
# CHECK: XFAIL: shtest-format :: xfail-target.txt
# CHECK: XFAIL: shtest-format :: xfail.txt
# CHECK: XPASS: shtest-format :: xpass.txt
# CHECK-NEXT: *** TEST 'shtest-format :: xpass.txt' FAILED ***
# CHECK-NEXT: Script
# CHECK-NEXT: --
# CHECK-NEXT: true
# CHECK-NEXT: --
# CHECK: Testing Time

# CHECK: Unexpected Passing Tests (1)
# CHECK: shtest-format :: xpass.txt

# CHECK: Failing Tests (3)
# CHECK: shtest-format :: external_shell/fail.txt
# CHECK: shtest-format :: external_shell/fail_with_bad_encoding.txt
# CHECK: shtest-format :: fail.txt

# CHECK: Expected Passes    : 7
# CHECK: Expected Failures  : 4
# CHECK: Unsupported Tests  : 5
# CHECK: Unresolved Tests   : 3
# CHECK: Unexpected Passes  : 1
# CHECK: Unexpected Failures: 3
