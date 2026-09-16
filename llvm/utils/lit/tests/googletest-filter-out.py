# Check that --filter-out applies to tests expanded from a GoogleTest shard.
# Filtering the failing and unresolved tests should leave a successful run.

# RUN: %{lit} --filter-out 'FirstTest/(subTestB|subTestD)' %{inputs}/googletest-format | FileCheck %s

# CHECK-NOT: Failed Tests
# CHECK-NOT: Unresolved Tests
# CHECK: Total Discovered Tests: 6
# CHECK-NEXT: {{ *}}Excluded: 2
# CHECK-NEXT: {{ *}}Skipped : 1
# CHECK-NEXT: {{ *}}Passed  : 3
# CHECK-NOT: {{Failed|Unresolved}}
