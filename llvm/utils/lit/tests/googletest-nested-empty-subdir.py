# Check that empty test_sub_dirs produces clean test names without double slashes
# for an executable found via lit's own directory recursion.

# RUN: not %{lit} -v --no-gtest-sharding %{inputs}/googletest-nested-empty-subdir | FileCheck %s

# CHECK: FAIL: googletest-nested-empty-subdir :: device/device-test.py

# CHECK: Failed Tests (1):
# CHECK-NEXT:   googletest-nested-empty-subdir :: FirstTest/subTestB
