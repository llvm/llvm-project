# Check the not command correctly handles POSIX signals

# UNSUPPORTED: system-windows

# RUN: not %{lit} -a %{inputs}/shtest-not-posix \
# RUN: | FileCheck -match-full-lines %s

# CHECK: -- Testing: 2 tests{{.*}}

# CHECK PASS: shtest-not-posix :: not-signal-crash.txt (1 of 2)

# CHECK: FAIL: shtest-not-posix :: not-signal.txt (2 of 2)
# CHECK: # error: command failed with exit status: 1
