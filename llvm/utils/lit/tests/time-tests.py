## Check that --skip-test-time-recording skips .lit_test_times.txt recording.

# RUN: %{lit-no-order-opt} --skip-test-time-recording %{inputs}/time-tests
# RUN: not ls %{inputs}/time-tests/.lit_test_times.txt

## Check that --time-tests (default 20 tests) generates a printed histogram.
# The slowest-test entries are matched with -DAG in any order to avoid
# performance-wise flakiness from relying on exact execution-time ordering.

# RUN: %{lit-no-order-opt} --time-tests %{inputs}/time-tests > %t.out
# RUN: FileCheck --check-prefix=DEFAULT < %t.out %s
# RUN: rm %{inputs}/time-tests/.lit_test_times.txt

# DEFAULT:      Slowest Tests (3 of 3):
# DEFAULT-DAG:  {{[0-9.]+}}s: time-tests :: a.txt
# DEFAULT-DAG:  {{[0-9.]+}}s: time-tests :: b.txt
# DEFAULT-DAG:  {{[0-9.]+}}s: time-tests :: c.txt
# DEFAULT:        Test Times (3):
# DEFAULT-NEXT: --------------------------------------------------------------------------
# DEFAULT-NEXT: [    Range    ] :: [               Percentage               ] :: [Count]
# DEFAULT-NEXT: --------------------------------------------------------------------------

## Check that --time-tests=1 limits the slowest-test list.

# RUN: %{lit-no-order-opt} --time-tests=1 %{inputs}/time-tests > %t.one.out
# RUN: FileCheck --check-prefix=ONE < %t.one.out %s
# RUN: rm %{inputs}/time-tests/.lit_test_times.txt

# ONE:       Slowest Tests (1 of 3):
# ONE-COUNT-1: {{[0-9.]+}}s: time-tests ::
# ONE:         Test Times (3):

## Check that --time-tests=all reports every timed test.

# RUN: %{lit-no-order-opt} --time-tests=all %{inputs}/time-tests > %t.all.out
# RUN: FileCheck --check-prefix=ALL < %t.all.out %s
# RUN: rm %{inputs}/time-tests/.lit_test_times.txt

# ALL:      Slowest Tests (3 of 3):
# ALL-DAG:  {{[0-9.]+}}s: time-tests :: a.txt
# ALL-DAG:  {{[0-9.]+}}s: time-tests :: b.txt
# ALL-DAG:  {{[0-9.]+}}s: time-tests :: c.txt
# ALL:        Test Times (3):

## Check that invalid --time-tests values are rejected.

# RUN: not %{lit-no-order-opt} --time-tests=0 %{inputs}/time-tests 2>&1 | FileCheck %s --check-prefix=INVALID

# INVALID: requires positive integer

## Check that malformed --time-tests values with extra '=' are rejected.

# RUN: not %{lit-no-order-opt} --time-tests=all=foo %{inputs}/time-tests 2>&1 | FileCheck %s --check-prefix=MALFORMED

# MALFORMED: requires positive integer, but found 'all=foo'
