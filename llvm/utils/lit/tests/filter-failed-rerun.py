# Checks that --filter-failed won't re-run tests that have passed
# since the last time --filter-failed was run.

# RUN: rm -rf %t
# RUN: cp -r %{inputs}%{fs-sep}filter-failed %t
#
# RUN: not %{lit} %t | FileCheck %s --check-prefix=CHECK-FIRST
#
# RUN: cp -f %t%{fs-sep}pass.txt %t%{fs-sep}fail.txt
# RUN: not %{lit} %t | FileCheck %s --check-prefix=CHECK-SECOND
# RUN: not %{lit} --filter-failed %t | FileCheck %s --check-prefix=CHECK-THIRD

# CHECK-FIRST: FAIL: filter-failed :: fail.txt

# CHECK-SECOND: PASS: filter-failed :: fail.txt

# CHECK-THIRD: Testing: 2 of 5 tests
# CHECK-THIRD-NOT: filter-failed :: fail.txt
