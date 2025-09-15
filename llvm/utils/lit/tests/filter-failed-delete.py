# Shows behaviour when a previously failed test was deleted
# before running with --filter-failed.

# RUN: not %{lit} %{inputs}/filter-failed-delete | FileCheck %s --check-prefix=CHECK-FIRST
#
# RUN: mv %{inputs}/filter-failed-delete/fail.txt %{inputs}/filter-failed-delete/fail.txt.bk
# RUN: not %{lit} --filter-failed %{inputs}/filter-failed-delete > %s.rerun.log
# RUN: mv %{inputs}/filter-failed-delete/fail.txt.bk %{inputs}/filter-failed-delete/fail.txt
#
# RUN: cat %s.rerun.log | FileCheck %s --check-prefix=CHECK-RERUN

# CHECK-FIRST: FAIL: filter-failed-delete :: fail.txt

# CHECK-RERUN-NOT: filter-failed-delete :: fail.txt
