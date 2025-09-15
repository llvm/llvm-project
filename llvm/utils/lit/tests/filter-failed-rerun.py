# Checks that --filter-failed won't re-run tests that have passed
# since the last time --filter-failed has run.

# RUN: not %{lit} %{inputs}/filter-failed-rerun | FileCheck %s --check-prefix=CHECK-FIRST
#
# RUN: mv %{inputs}/filter-failed-rerun/fail.txt %{inputs}/filter-failed-rerun/fail.txt.bk
# RUN: cp %{inputs}/filter-failed-rerun/pass.txt %{inputs}/filter-failed-rerun/fail.txt
# RUN: not %{lit} %{inputs}/filter-failed-rerun > %s.rerun-1.log
# RUN: not %{lit} --filter-failed %{inputs}/filter-failed-rerun > %s.rerun-2.log
# RUN: mv %{inputs}/filter-failed-rerun/fail.txt.bk %{inputs}/filter-failed-rerun/fail.txt
#
# RUN: cat %s.rerun-1.log | FileCheck %s --check-prefix=CHECK-RERUN1
# RUN: cat %s.rerun-2.log | FileCheck %s --check-prefix=CHECK-RERUN2

# CHECK-FIRST: FAIL: filter-failed-rerun :: fail.txt

# CHECK-RERUN1: PASS: filter-failed-rerun :: fail.txt

# CHECK-RERUN2: Testing: 1 of 3 tests
# CHECK-RERUN2-NOT: filter-failed-rerun :: fail.txt
