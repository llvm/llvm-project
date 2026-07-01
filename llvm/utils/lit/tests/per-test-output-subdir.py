# Check that --per-test-output-subdir isolates a run's writable output tree by
# splicing the caller-supplied ID as a component of test_exec_root, so that
# %t/%T/Output resolve under test_exec_root/<ID>. This lets multiple lit runs
# use the same tests concurrently against one build tree without clobbering each
# other, while reusing an ID reuses the tree.

# With the option, the temp path is nested under the given ID directory.
# RUN: rm -rf %t && mkdir -p %t
# RUN: %{lit} -a --per-test-output-subdir run-abc123 -Dexec_root=%t \
# RUN:     %{inputs}/per-test-output-subdir | \
# RUN:   FileCheck --check-prefix=ISOLATED %s

# Without the option, the temp path is not nested under the ID directory.
# RUN: rm -rf %t && mkdir -p %t
# RUN: %{lit} -a -Dexec_root=%t \
# RUN:     %{inputs}/per-test-output-subdir | \
# RUN:   FileCheck --check-prefix=SHARED %s

# ISOLATED: TEMP_PATH={{.*}}/run-abc123/{{.*}}Output

# SHARED: TEMP_PATH=
# SHARED-NOT: /run-abc123
