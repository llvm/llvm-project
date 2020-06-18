# Check that we can inject commands at the beginning of a ShTest.

# RUN: %{lit} -j 1 %{inputs}/shtest-inject/test-empty.txt --show-all | FileCheck --check-prefix=CHECK-TEST1 %s
#
# CHECK-TEST1: Script:
# CHECK-TEST1: --
# CHECK-TEST1: echo "THIS WAS"
# CHECK-TEST1: echo "INJECTED"
# CHECK-TEST1: --
#
# CHECK-TEST1: THIS WAS
# CHECK-TEST1: INJECTED
#
# CHECK-TEST1: Passed: 1

# RUN: %{lit} -j 1 %{inputs}/shtest-inject/test-one.txt --show-all | FileCheck --check-prefix=CHECK-TEST2 %s
#
# CHECK-TEST2: Script:
# CHECK-TEST2: --
# CHECK-TEST2: echo "THIS WAS"
# CHECK-TEST2: echo "INJECTED"
# CHECK-TEST2: echo "IN THE FILE"
# CHECK-TEST2: --
#
# CHECK-TEST2: THIS WAS
# CHECK-TEST2: INJECTED
# CHECK-TEST2: IN THE FILE
#
# CHECK-TEST2: Passed: 1

# RUN: %{lit} -j 1 %{inputs}/shtest-inject/test-many.txt --show-all | FileCheck --check-prefix=CHECK-TEST3 %s
#
# CHECK-TEST3: Script:
# CHECK-TEST3: --
# CHECK-TEST3: echo "THIS WAS"
# CHECK-TEST3: echo "INJECTED"
# CHECK-TEST3: echo "IN THE FILE"
# CHECK-TEST3: echo "IF IT WORKS"
# CHECK-TEST3: echo "AS EXPECTED"
# CHECK-TEST3: --
#
# CHECK-TEST3: THIS WAS
# CHECK-TEST3: INJECTED
# CHECK-TEST3: IN THE FILE
# CHECK-TEST3: IF IT WORKS
# CHECK-TEST3: AS EXPECTED
#
# CHECK-TEST3: Passed: 1
