# Check the behavior of --max-failures option.
#
# RUN: not %{lit}                  -j 1 %{inputs}/max-failures >  %t.out 2>&1
# RUN: not %{lit} --max-failures=1 -j 1 %{inputs}/max-failures >> %t.out 2>&1
# RUN: not %{lit} --max-failures=2 -j 1 %{inputs}/max-failures >> %t.out 2>&1
# RUN: not %{lit} --max-failures=0 -j 1 %{inputs}/max-failures 2>> %t.out
# RUN: FileCheck < %t.out %s
#

# CHECK-NOT: reached maximum number of test failures
# CHECK-NOT: Skipped Tests
# CHECK: Unexpected Failures: 3

# CHECK: reached maximum number of test failures, skipping remaining tests
# CHECK: Skipped Tests      : 2
# CHECK: Unexpected Failures: 1

# CHECK: reached maximum number of test failures, skipping remaining tests
# CHECK: Skipped Tests      : 1
# CHECK: Unexpected Failures: 2

# CHECK: error: argument --max-failures: requires positive integer, but found '0'
