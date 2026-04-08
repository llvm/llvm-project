# Shows behaviour when a previously failed test was deleted
# before running with --filter-failed.

# RUN: rm -rf %t
# RUN: cp -r %{inputs}%{fs-sep}filter-failed %t
#
# RUN: not %{lit} %t | FileCheck %s --check-prefix=CHECK-FIRST
#
# RUN: rm %t%{fs-sep}fail.txt
# RUN: not %{lit} --filter-failed %t | FileCheck %s --check-prefix=CHECK-SECOND

# CHECK-FIRST: Testing: 5 tests
# CHECK-FIRST: FAIL: filter-failed :: fail.txt

# CHECK-SECOND: Testing: 2 of 4 tests
# CHECK-SECOND-NOT: filter-failed :: fail.txt
