# Checks that --filter-failed only runs tests that previously failed.

# RUN: not %{lit} %{inputs}/filter-failed
#
# RUN: rm -f %{inputs}/filter-failed/new-fail.txt
# RUN: echo "RUN: false" > %{inputs}/filter-failed/new-fail.txt
#
# RUN: rm -f %{inputs}/filter-failed/new-pass.txt
# RUN: echo "RUN: true" > %{inputs}/filter-failed/new-pass.txt
#
# RUN: not %{lit} --filter-failed %{inputs}/filter-failed | FileCheck %s

# END.

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
