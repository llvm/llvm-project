## Check that we can reorder test runs.

# RUN: cp %{inputs}/reorder/.lit_test_times.txt %{inputs}/reorder/.lit_test_times.txt.orig
# RUN: %{lit} -j1 %{inputs}/reorder > %t.out
# RUN: cp %{inputs}/reorder/.lit_test_times.txt %{inputs}/reorder/.lit_test_times.txt.new
# RUN: cp %{inputs}/reorder/.lit_test_times.txt.orig %{inputs}/reorder/.lit_test_times.txt
# RUN: not diff %{inputs}/reorder/.lit_test_times.txt.new %{inputs}/reorder/.lit_test_times.txt.orig
# RUN: FileCheck --check-prefix=TIMES --implicit-check-not= < %{inputs}/reorder/.lit_test_times.txt.new %s
# RUN: FileCheck < %t.out %s
# END.

# TIMES: not-executed.txt
# TIMES-NEXT: subdir/ccc.txt
# TIMES-NEXT: bbb.txt
# TIMES-NEXT: aaa.txt
# TIMES-NEXT: new-test.txt

# CHECK:     -- Testing: 4 tests, 1 workers --
# CHECK-NEXT: PASS: reorder :: subdir/ccc.txt
# CHECK-NEXT: PASS: reorder :: bbb.txt
# CHECK-NEXT: PASS: reorder :: aaa.txt
# CHECK-NEXT: PASS: reorder :: new-test.txt
# CHECK:     Passed: 4
