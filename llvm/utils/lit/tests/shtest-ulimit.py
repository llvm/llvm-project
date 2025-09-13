# Check the ulimit command

# ulimit does not work on non-POSIX platforms.
# UNSUPPORTED: system-windows

# RUN: not %{lit} -a -v %{inputs}/shtest-ulimit | FileCheck %s
# RUN: %if system-linux %{ not %{lit} -a -v %{inputs}/shtest-ulimit-linux | FileCheck %s --check-prefix CHECK-LINUX %}

# CHECK: -- Testing: 2 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit-bad-arg.txt ({{[^)]*}})
# CHECK: ulimit -n
# CHECK: 'ulimit' requires two arguments

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit_okay.txt ({{[^)]*}})
# CHECK: ulimit -n 50
# CHECK: RLIMIT_NOFILE=50

# CHECK-LINUX: -- Testing: 1 tests{{.*}}

# CHECK-LINUX-LABEL: FAIL: shtest-ulimit :: ulimit_okay.txt ({{[^)]*}})
# CHECK-LINUX: ulimit -v 1048576
# CHECK-LINUX: RLIMIT_AS=1073741824
