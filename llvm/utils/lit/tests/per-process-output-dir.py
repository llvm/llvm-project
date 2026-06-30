# Check that --per-process-output-dir isolates each lit process's writable
# output tree by splicing a unique "pid-<pid>" component into test_exec_root,
# so that %t/%T/Output resolve under a per-process directory. This lets multiple
# lit processes run the same tests concurrently against one build tree without
# clobbering each other.

# With the flag, the temp path is nested under a pid-<pid> directory.
# RUN: rm -rf %t && mkdir -p %t
# RUN: %{lit} -a --per-process-output-dir -Dexec_root=%t \
# RUN:     %{inputs}/per-process-output-dir | \
# RUN:   FileCheck --check-prefix=ISOLATED %s

# Without the flag, the temp path is not nested under a pid- directory.
# RUN: rm -rf %t && mkdir -p %t
# RUN: %{lit} -a -Dexec_root=%t \
# RUN:     %{inputs}/per-process-output-dir | \
# RUN:   FileCheck --check-prefix=SHARED %s

# ISOLATED: TEMP_PATH={{.*}}/pid-{{[0-9]+}}/{{.*}}Output

# SHARED: TEMP_PATH=
# SHARED-NOT: /pid-
