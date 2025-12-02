# Check the ulimit command

# ulimit does not work on non-POSIX platforms.
# Solaris for some reason does not respect ulimit -n, so mark it unsupported
# as well.
# UNSUPPORTED: system-windows, system-cygwin, system-solaris

# RUN: %{python} %S/Inputs/shtest-ulimit/print_limits.py | grep RLIMIT_NOFILE \
# RUN:   | sed -n -e 's/.*=//p' | tr -d '\n' > %t.nofile_limit

# RUN: not %{lit} -v %{inputs}/shtest-ulimit --order=lexical \
# RUN:   | FileCheck -DBASE_NOFILE_LIMIT=%{readfile:%t.nofile_limit} %s

# CHECK: -- Testing: 3 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit-bad-arg.txt ({{[^)]*}})
# CHECK: ulimit -n
# CHECK: 'ulimit' requires two arguments

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit_okay.txt ({{[^)]*}})
# CHECK: ulimit -n 50
# CHECK: ulimit -f 5
# CHECK: RLIMIT_NOFILE=50
# CHECK: RLIMIT_FSIZE=5

# CHECK-LABEL: FAIL: shtest-ulimit :: ulimit_reset.txt ({{[^)]*}})
# CHECK: RLIMIT_NOFILE=[[BASE_NOFILE_LIMIT]]
