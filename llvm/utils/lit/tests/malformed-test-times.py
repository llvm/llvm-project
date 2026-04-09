## Check that malformed .lit_test_times.txt lines do not crash discovery.
##
## The valid timing entries should still be honored for smart ordering.

# RUN: cp %{inputs}/malformed-test-times/lit_test_times %{inputs}/malformed-test-times/.lit_test_times.txt
# RUN: %{lit-no-order-opt} %{inputs}/malformed-test-times > %t.out
# RUN: FileCheck < %t.out %s

# CHECK: -- Testing: 2 tests, 1 workers --
# CHECK-NEXT: PASS: malformed-test-times :: b.txt
# CHECK-NEXT: PASS: malformed-test-times :: a.txt
# CHECK: Passed: 2
