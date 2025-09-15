# Check the ulimit command

# ulimit does not work on non-POSIX platforms.
# Solaris for some reason does not respect ulimit -n, so mark it unsupported
# as well.
# UNSUPPORTED: system-windows, system-solaris

# RUN: not %{lit} -a -v %{inputs}/shtest-ulimit | FileCheck %s

# CHECK: -- Testing: 2 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit-bad-arg.txt ({{[^)]*}})
# CHECK: ulimit -n
# CHECK: 'ulimit' requires two arguments

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit_okay.txt ({{[^)]*}})
# CHECK: ulimit -n 50
# CHECK: RLIMIT_NOFILE=50
