# Check that --test-output-root relocates each suite's writable output tree to
# <root>/<suite-name>, so %t/%T/Output resolve under the given root directory
# instead of the suite's build test_exec_root. This lets test output live
# outside the build tree and lets multiple lit runs share one build tree without
# clobbering each other; reusing a root reuses the tree.

# With the option, the temp path is under <root>/<suite-name>, not the build dir.
# RUN: rm -rf %t && mkdir -p %t/build %t/out
# RUN: %{lit} -a --test-output-root %t/out -Dexec_root=%t/build \
# RUN:     %{inputs}/test-output-root | \
# RUN:   FileCheck --check-prefix=ROOTED %s

# Without the option, the temp path stays under the build test_exec_root.
# RUN: rm -rf %t && mkdir -p %t/build
# RUN: %{lit} -a -Dexec_root=%t/build \
# RUN:     %{inputs}/test-output-root | \
# RUN:   FileCheck --check-prefix=DEFAULT %s

# ROOTED: TEMP_PATH={{.*}}/out/output-root-suite/{{.*}}Output
# ROOTED-NOT: TEMP_PATH={{.*}}/build/

# DEFAULT: TEMP_PATH={{.*}}/build/{{.*}}Output
# DEFAULT-NOT: output-root-suite
