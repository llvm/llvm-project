# Check the various features of the GoogleTest format.

# RUN: not %{lit} -v %{inputs}/googletest-upstream-format > %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK: -- Testing:
# CHECK: PASS: googletest-upstream-format :: [[PATH:[Dd]ummy[Ss]ub[Dd]ir/]][[FILE:OneTest\.py]]/FirstTest.subTestA
# CHECK: FAIL: googletest-upstream-format :: [[PATH]][[FILE]]/[[TEST:FirstTest\.subTestB]]
# CHECK-NEXT: *** TEST 'googletest-upstream-format :: [[PATH]][[FILE]]/[[TEST]]' FAILED ***
# CHECK-NEXT: Script:
# CHECK-NEXT: --
# CHECK-NEXT: [[FILE]] --gtest_filter=[[TEST]]
# CHECK-NEXT: --
# CHECK-NEXT: Running main() from gtest_main.cc
# CHECK-NEXT: I am subTest B, I FAIL
# CHECK-NEXT: And I have two lines of output
# CHECK: SKIPPED: googletest-upstream-format :: [[PATH]][[FILE]]/FirstTest.subTestC
# CHECK: UNRESOLVED: googletest-upstream-format :: [[PATH]][[FILE]]/[[TEST:FirstTest\.subTestD]]
# CHECK-NEXT: *** TEST 'googletest-upstream-format :: [[PATH]][[FILE]]/[[TEST]]' FAILED ***
# CHECK-NEXT: Script:
# CHECK-NEXT: --
# CHECK-NEXT: [[FILE]] --gtest_filter=[[TEST]]
# CHECK-NEXT: --
# CHECK-NEXT: Unable to find [ PASSED ] 1 test. in gtest output
# CHECK: I am subTest D, I am UNRESOLVED
# CHECK: ***
# CHECK: PASS: googletest-upstream-format :: [[PATH]][[FILE]]/ParameterizedTest/0.subTest
# CHECK: PASS: googletest-upstream-format :: [[PATH]][[FILE]]/ParameterizedTest/1.subTest
# CHECK: Failed Tests (1)
# CHECK: Skipped{{ *}}: 1
# CHECK: Passed{{ *}}: 3
# CHECK: Unresolved{{ *}}: 1
# CHECK: Failed{{ *}}: 1
