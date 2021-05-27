# Check the various features of the GoogleTest format.

# FIXME: this test depends on order of tests
# RUN: rm -f %{inputs}/googletest-format/.lit_test_times.txt

# RUN: not %{lit} -j 1 -v %{inputs}/googletest-format > %t.out
# FIXME: Temporarily dump test output so we can debug failing tests on
# buildbots.
# RUN: cat %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK: -- Testing:
# CHECK: PASS: googletest-format :: [[PATH:[Dd]ummy[Ss]ub[Dd]ir/]][[FILE:OneTest\.py]]/FirstTest.subTestA
# CHECK: FAIL: googletest-format :: [[PATH]][[FILE]]/[[TEST:FirstTest\.subTestB]]
# CHECK-NEXT: *** TEST 'googletest-format :: [[PATH]][[FILE]]/[[TEST]]' FAILED ***
# CHECK-NEXT: Script:
# CHECK-NEXT: --
# CHECK-NEXT: [[FILE]] --gtest_filter=[[TEST]]
# CHECK-NEXT: --
# CHECK-NEXT: I am subTest B, I FAIL
# CHECK-NEXT: And I have two lines of output
# CHECK: ***
# CHECK: SKIPPED: googletest-format :: [[PATH]][[FILE]]/FirstTest.subTestC
# CHECK: UNRESOLVED: googletest-format :: [[PATH]][[FILE]]/[[TEST:FirstTest\.subTestD]]
# CHECK-NEXT: *** TEST 'googletest-format :: [[PATH]][[FILE]]/[[TEST]]' FAILED ***
# CHECK-NEXT: Script:
# CHECK-NEXT: --
# CHECK-NEXT: [[FILE]] --gtest_filter=[[TEST]]
# CHECK-NEXT: --
# CHECK-NEXT: Unable to find [ PASSED ] 1 test. in gtest output
# CHECK: I am subTest D, I am UNRESOLVED
# CHECK: PASS: googletest-format :: [[PATH]][[FILE]]/ParameterizedTest/0.subTest
# CHECK: PASS: googletest-format :: [[PATH]][[FILE]]/ParameterizedTest/1.subTest
# CHECK: Failed Tests (1)
# CHECK: Skipped{{ *}}: 1
# CHECK: Passed{{ *}}: 3
# CHECK: Unresolved{{ *}}: 1
# CHECK: Failed{{ *}}: 1
