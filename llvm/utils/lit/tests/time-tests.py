## Check that --skip-test-time-recording skips .lit_test_times.txt recording.

# RUN: %{lit-no-order-opt} --skip-test-time-recording %{inputs}/time-tests
# RUN: not ls %{inputs}/time-tests/.lit_test_times.txt > %t.out 2>&1
# RUN: FileCheck --check-prefix=CHECK-NOFILE < %t.out %s

## Check that --time-tests generates a printed histogram

# RUN: %{lit-no-order-opt} --time-tests %{inputs}/time-tests > %t.out
# RUN: FileCheck < %t.out %s
# RUN: rm %{inputs}/time-tests/.lit_test_times.txt

# CHECK-NOFILE: cannot access 'Inputs/time-tests/.lit_test_times.txt': No such file or directory

# CHECK:      Tests Times:
# CHECK-NEXT: --------------------------------------------------------------------------
# CHECK-NEXT: [    Range    ] :: [               Percentage               ] :: [Count]
# CHECK-NEXT: --------------------------------------------------------------------------
