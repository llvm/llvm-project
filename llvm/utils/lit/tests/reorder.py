## Check that we can reorder test runs.

# RUN: cp %{inputs}/reorder/.lit_test_times.txt %{inputs}/reorder/.lit_test_times.txt.orig
# RUN: %{lit} -j1 %{inputs}/reorder | FileCheck %s
# RUN: not diff %{inputs}/reorder/.lit_test_times.txt %{inputs}/reorder/.lit_test_times.txt.orig
# UNSUPPORTED: system-windows
# END.

# CHECK:     -- Testing: 3 tests, 1 workers --
# CHECK-NEXT: PASS: reorder :: subdir/ccc.txt
# CHECK-NEXT: PASS: reorder :: bbb.txt
# CHECK-NEXT: PASS: reorder :: aaa.txt
# CHECK:     Passed: 3
