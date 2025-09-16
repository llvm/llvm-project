# Shows behaviour when a previously failed test was deleted
# before running with --filter-failed.

# RUN: rm -rf %t
# RUN: cp -r %{inputs}/filter-failed %t
#
# RUN: not %{lit} %t | FileCheck %s --check-prefix=CHECK-FIRST
#
# RUN: rm %t/fail.txt
# RUN: not %{lit} --filter-failed %t | FileCheck %s --check-prefix=CHECK-RERUN

# CHECK-FIRST: FAIL: filter-failed :: fail.txt

# CHECK-RERUN-NOT: filter-failed :: fail.txt
