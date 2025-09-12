# Check the ulimit command

# ulimit does not work on non-POSIX platforms.
# UNSUPPORTED: system-windows

# RUN: not %{lit} -a -v %{inputs}/shtest-ulimit | FileCheck %s

# CHECK: -- Testing: 2 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit-bad-arg.txt ({{[^)]*}})
# CHECK: ulimit -n
# CHECK: 'ulimit' requires two arguments

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit_okay.txt ({{[^)]*}})
# CHECK: ulimit -v 1048576
# CHECK: ulimit -n 50
# CHECK: RLIMIT_AS=1073741824
# CHECK: RLIMIT_NOFILE=50
