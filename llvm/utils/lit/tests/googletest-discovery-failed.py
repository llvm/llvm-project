# Check for correct error message when discovery of tests fails.
#
# RUN: not %{lit} -j 1 -v %{inputs}/googletest-discovery-failed > %t.cmd.out
# RUN: FileCheck < %t.cmd.out %s


# CHECK: -- Testing:
# CHECK: Failing Tests (1):
# CHECK:   googletest-discovery-failed :: subdir/OneTest.py/failed_to_discover_tests_from_gtest
# CHECK: Unexpected Failures: 1
