# Check that --test-output-root relocates each suite's writable output tree to
# <root>/<suite-name>, so %t/%T/Output resolve under the given root directory
# instead of the suite's build test_exec_root. This lets test output live
# outside the build tree and lets multiple lit runs share one build tree without
# clobbering each other; reusing a root reuses the tree.
#
# The passed exec root uses a distinctive "execroot" directory name (rather than
# "build") so the ROOTED-NOT check below can't be fooled by an ambient "/build/"
# segment in %t (e.g. when the lit test tree itself lives under a build dir).

# With the option, the temp path is under <root>/<suite-name>, not the exec root.
# RUN: rm -rf %t && mkdir -p %t/execroot %t/out
# RUN: %{lit} -a --test-output-root %t/out -Dexec_root=%t/execroot \
# RUN:     %{inputs}/test-output-root | \
# RUN:   FileCheck --check-prefix=ROOTED %s

# Without the option, the temp path stays under the exec root.
# RUN: rm -rf %t && mkdir -p %t/execroot
# RUN: %{lit} -a -Dexec_root=%t/execroot \
# RUN:     %{inputs}/test-output-root | \
# RUN:   FileCheck --check-prefix=DEFAULT %s

# ROOTED: TEMP_PATH={{.*}}/out/output-root-suite/{{.*}}Output
# ROOTED-NOT: TEMP_PATH={{.*}}/execroot/

# DEFAULT: TEMP_PATH={{.*}}/execroot/{{.*}}Output
# DEFAULT-NOT: output-root-suite
