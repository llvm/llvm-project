## Check that we can run tests early.

# RUN: %{lit} -j1 %{inputs}/early-tests | FileCheck %s

# CHECK:     -- Testing: 3 tests, 1 workers --
# CHECK-NEXT: PASS: early-tests :: subdir/ccc.txt
# CHECK-NEXT: PASS: early-tests :: aaa.txt
# CHECK-NEXT: PASS: early-tests :: bbb.txt
# CHECK:     Passed: 3
