# Check the behavior of --max-failures option.
#
# RUN: not %{lit} -j 1 -v %{inputs}/max-failures > %t.out
# RUN: not %{lit} --max-failures=1 -j 1 -v %{inputs}/max-failures >> %t.out
# RUN: not %{lit} --max-failures=2 -j 1 -v %{inputs}/max-failures >> %t.out
# RUN: not %{lit} --max-failures=0 -j 1 -v %{inputs}/max-failures 2>> %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK: Failing Tests (27)
# CHECK: Failing Tests (1)
# CHECK: Failing Tests (2)
# CHECK: error: Option '--max-failures' requires positive integer
