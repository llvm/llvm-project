# Check the ulimit command

# ulimit does not work on non-POSIX platforms.
# These tests are specific to options that Darwin does not support.
# UNSUPPORTED: system-windows, system-darwin

# RUN: not %{lit} -a -v %{inputs}/shtest-ulimit-nondarwin | FileCheck %s

# CHECK: -- Testing: 1 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit_okay.txt ({{[^)]*}})
# CHECK: ulimit -v 1048576
# CHECK: RLIMIT_AS=1073741824
