# Check that empty test_sub_dirs produces clean test names without double slashes.

# RUN: not %{lit} -v --no-gtest-sharding %{inputs}/googletest-empty-subdir | FileCheck %s

# CHECK: FAIL: googletest-empty-subdir :: device-test.py
# CHECK: *** TEST 'googletest-empty-subdir :: device-test.py' FAILED ***

# CHECK: Failed Tests (1):
# CHECK-NEXT:   googletest-empty-subdir :: FirstTest/subTestB
