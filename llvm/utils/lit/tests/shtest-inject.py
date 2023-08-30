# Check that we can inject commands at the beginning of a ShTest.

# RUN: %{lit} %{inputs}/shtest-inject/test-empty.txt --show-all | FileCheck --check-prefix=CHECK-TEST1 %s
#
# CHECK-TEST1: THIS WAS
# CHECK-TEST1: INJECTED
#
# CHECK-TEST1: Passed: 1

# RUN: %{lit} %{inputs}/shtest-inject/test-one.txt --show-all | FileCheck --check-prefix=CHECK-TEST2 %s
#
# CHECK-TEST2: THIS WAS
# CHECK-TEST2: INJECTED
# CHECK-TEST2: IN THE FILE
#
# CHECK-TEST2: Passed: 1

# RUN: %{lit} %{inputs}/shtest-inject/test-many.txt --show-all | FileCheck --check-prefix=CHECK-TEST3 %s
#
# CHECK-TEST3: THIS WAS
# CHECK-TEST3: INJECTED
# CHECK-TEST3: IN THE FILE
# CHECK-TEST3: IF IT WORKS
# CHECK-TEST3: AS EXPECTED
#
# CHECK-TEST3: Passed: 1
