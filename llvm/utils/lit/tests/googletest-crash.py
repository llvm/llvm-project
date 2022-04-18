# Check GoogleTest shard test crashes are handled.

# RUN: not %{lit} -v %{inputs}/googletest-crash | FileCheck %s

# CHECK: -- Testing:
# CHECK: FAIL: googletest-crash :: [[PATH:[Dd]ummy[Ss]ub[Dd]ir/]][[FILE:OneTest\.py]]/0
# CHECK: *** TEST 'googletest-crash :: [[PATH]][[FILE]]/0{{.*}} FAILED ***
# CHECK-NEXT: Script(shard):
# CHECK-NEXT: --
# CHECK-NEXT: GTEST_COLOR=no
# CHECK-NEXT: GTEST_SHUFFLE=0
# CHECK-NEXT: GTEST_TOTAL_SHARDS=6
# CHECK-NEXT: GTEST_SHARD_INDEX=0
# CHECK-NEXT: GTEST_OUTPUT=json:[[JSON:.*\.json]]
# CHECK-NEXT: [[FILE]]
# CHECK-NEXT: --
# CHECK-NEXT: shard JSON output does not exist: [[JSON]]
# CHECK-NEXT: ***
# CHECK: Failed Tests (1):
# CHECK-NEXT:   googletest-crash :: [[PATH]][[FILE]]/0/6
# CHECK: Failed{{ *}}: 1
