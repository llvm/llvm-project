# Checks that --filter-failed only runs tests that previously failed.

# RUN: not %{lit} %{inputs}/ignore-fail
# RUN: not %{lit} --filter-failed %{inputs}/ignore-fail | FileCheck %s

# END.

# CHECK: Testing: 3 of 5 tests
# CHECK-DAG: FAIL: ignore-fail :: fail.txt
# CHECK-DAG: UNRESOLVED: ignore-fail :: unresolved.txt
# CHECK-DAG: XPASS: ignore-fail :: xpass.txt

# CHECK: Testing Time:
# CHECK: Total Discovered Tests:
# CHECK-NEXT:   Excluded : 2 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NEXT:   Unresolved : 1 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NEXT:   Failed : 1 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NEXT:   Unexpectedly Passed: 1 {{\([0-9]*\.[0-9]*%\)}}
