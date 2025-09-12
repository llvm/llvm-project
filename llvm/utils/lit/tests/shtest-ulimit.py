# Check the ulimit command

# ulimit does not work on non-POSIX platforms.
# UNSUPPORTED: system-windows

# TODO(boomanaiden154): The test fails on some non-Linux POSIX
# platforms (like MacOS) due to the underlying system not supporting
# ulimit -v. This test needs to be carved up so we keep full test
# coverage on Linux and as much as possible on other platforms.
# REQUIRES: system-linux

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
