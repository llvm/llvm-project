# RUN: not %{lit} -v --no-gtest-sharding %{inputs}/googletest-nested-empty-subdir | FileCheck %s

# CHECK: FAIL: googletest-nested-empty-subdir :: device/device-test.py

# CHECK: Failed Tests (1):
# CHECK-NEXT:   googletest-nested-empty-subdir :: FirstTest/subTestB

# RUN: not %{lit} -v %{inputs}/googletest-nested-empty-subdir | FileCheck %s --check-prefix=SHARD

# SHARD: FAIL: googletest-nested-empty-subdir :: device/device-test.py/{{[0-9]+}}/{{[0-9]+}}
# SHARD-NOT: device//
