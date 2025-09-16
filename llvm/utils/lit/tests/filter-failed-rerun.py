# Checks that --filter-failed won't re-run tests that have passed
# since the last time --filter-failed has run.

# RUN: rm -rf %t
# RUN: cp -r %{inputs}/filter-failed %t
#
# RUN: not %{lit} %t | FileCheck %s --check-prefix=CHECK-FIRST
#
# RUN: cp %t/pass.txt %t/fail.txt
# RUN: not %{lit} %t | FileCheck %s --check-prefix=CHECK-RERUN1
# RUN: not %{lit} --filter-failed %t | FileCheck %s --check-prefix=CHECK-RERUN2

# CHECK-FIRST: FAIL: filter-failed :: fail.txt

# CHECK-RERUN1: PASS: filter-failed :: fail.txt

# CHECK-RERUN2: Testing: 2 of 5 tests
# CHECK-RERUN2-NOT: filter-failed :: fail.txt
