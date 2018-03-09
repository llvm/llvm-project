# Check the various features of the GoogleTest format.
#
# RUN: not %{lit} -j 1 -v %{inputs}/googletest-upstream-format > %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK: -- Testing:
# CHECK: PASS: googletest-upstream-format :: {{[Dd]ummy[Ss]ub[Dd]ir}}/OneTest.py/FirstTest.subTestA
# CHECK: FAIL: googletest-upstream-format :: {{[Dd]ummy[Ss]ub[Dd]ir}}/OneTest.py/FirstTest.subTestB
# CHECK-NEXT: *** TEST 'googletest-upstream-format :: {{[Dd]ummy[Ss]ub[Dd]ir}}/OneTest.py/FirstTest.subTestB' FAILED ***
# CHECK-NEXT: Running main() from gtest_main.cc
# CHECK-NEXT: I am subTest B, I FAIL
# CHECK-NEXT: And I have two lines of output
# CHECK: ***
# CHECK: PASS: googletest-upstream-format :: {{[Dd]ummy[Ss]ub[Dd]ir}}/OneTest.py/ParameterizedTest/0.subTest
# CHECK: PASS: googletest-upstream-format :: {{[Dd]ummy[Ss]ub[Dd]ir}}/OneTest.py/ParameterizedTest/1.subTest
# CHECK: Failing Tests (1)
# CHECK: Expected Passes    : 3
# CHECK: Unexpected Failures: 1
