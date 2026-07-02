# Check the output is expected when tests pass but sanitizer fails.
# Note that there is only one shard which has only one sub-test (subTestA). The shard
# fails with a non-zero exit code (sanitizer error) even though the sub-test passes.
# The summary shows only the shard-level result (Failed: 1).

# RUN: not %{lit} -v --order=random %{inputs}/googletest-sanitizer-error > %t.out
# FIXME: Temporarily dump test output so we can debug failing tests on
# buildbots.
# RUN: cat %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK: -- Testing:
# CHECK: FAIL: googletest-sanitizer-error :: [[PATH:[Dd]ummy[Ss]ub[Dd]ir/]][[FILE:OneTest\.py]]/0
# CHECK: *** TEST 'googletest-sanitizer-error :: [[PATH]][[FILE]]/0{{.*}} FAILED ***
# CHECK-NEXT: Script(shard):
# CHECK-NEXT: --
# CHECK-NEXT: GTEST_OUTPUT=json:{{[^[:space:]]*}} GTEST_SHUFFLE=1 GTEST_TOTAL_SHARDS={{[1-6]}} GTEST_SHARD_INDEX=0 GTEST_RANDOM_SEED=123 {{.*}}[[FILE]]
# CHECK-NEXT: --
# CHECK-EMPTY:
# CHECK-EMPTY:
# CHECK:      [ RUN      ] FirstTest.subTestA
# CHECK-NEXT: [       OK ] FirstTest.subTestA (8 ms)
# CHECK:      --
# CHECK-NEXT: exit: 1
# CHECK-NEXT: --
# CHECK:      Failed Tests (1):
# CHECK-NEXT:   googletest-sanitizer-error :: [[PATH]][[FILE]]/0/{{[0-9]+}}
# CHECK: Failed{{ *}}: 1
