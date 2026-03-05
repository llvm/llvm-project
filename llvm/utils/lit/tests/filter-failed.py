# Checks that --filter-failed only runs tests that previously failed.

# RUN: rm -rf %t
# RUN: cp -r %{inputs}%{fs-sep}filter-failed %t
#
# RUN: not %{lit} %t
#
# RUN: echo "RUN: false" > %t%{fs-sep}new-fail.txt
# RUN: echo "RUN: true"  > %t%{fs-sep}new-pass.txt
#
# RUN: not %{lit} --filter-failed %t | FileCheck %s

# CHECK: Testing: 3 of 7 tests
# CHECK-DAG: FAIL: filter-failed :: fail.txt
# CHECK-DAG: UNRESOLVED: filter-failed :: unresolved.txt
# CHECK-DAG: XPASS: filter-failed :: xpass.txt

# CHECK: Testing Time:
# CHECK: Total Discovered Tests:
# CHECK-NEXT:   Excluded : 4 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NEXT:   Unresolved : 1 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NEXT:   Failed : 1 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NEXT:   Unexpectedly Passed: 1 {{\([0-9]*\.[0-9]*%\)}}
