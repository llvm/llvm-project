# Check the ulimit command

# ulimit does not work on non-POSIX platforms.
# These tests are specific to options that Darwin does not support.
# UNSUPPORTED: system-windows, system-cygwin, system-darwin, system-aix, system-solaris

# RUN: not %{lit} -v %{inputs}/shtest-ulimit-nondarwin | FileCheck %s

# CHECK: -- Testing: 2 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit_okay.txt ({{[^)]*}})
# CHECK: ulimit -v 1048576
# CHECK: ulimit -s 256
# CHECK: RLIMIT_AS=1073741824
# CHECK: RLIMIT_STACK=262144

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit_unlimited.txt ({{[^)]*}})
# CHECK: ulimit -f 5
# CHECK: RLIMIT_FSIZE=5
# CHECK: ulimit -f unlimited
# CHECK: RLIMIT_FSIZE=-1
