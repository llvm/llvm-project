# RUN: not %{lit} -v --no-gtest-sharding %{inputs}/googletest-empty-subdir | FileCheck %s

# CHECK: FAIL: googletest-empty-subdir :: device-test.py
# CHECK: *** TEST 'googletest-empty-subdir :: device-test.py' FAILED ***

# CHECK: Failed Tests (1):
# CHECK-NEXT:   googletest-empty-subdir :: FirstTest/subTestB

# RUN: not %{lit} -v %{inputs}/googletest-empty-subdir | FileCheck %s --check-prefix=SHARD

# SHARD: FAIL: googletest-empty-subdir :: device-test.py/{{[0-9]+}}/{{[0-9]+}}
# SHARD-NOT: :: /
