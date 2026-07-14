# Check that --test-output-root relocates each suite's writable output tree to
# <root>/<suite-name>, so %t/Output/etc. resolve under the given root directory
# instead of the suite's build test_exec_root.
#
# Both cases capture the outer lit's %t as TESTDIR and match the inner temp path
# against it exactly, so each check fully pins where the output landed.

# With the option, the temp path is under <root>/<suite-name>, not the exec root.
# RUN: rm -rf %t && mkdir -p %t/execroot %t/out
# RUN: %{lit} -a --test-output-root %t/out -Dexec_root=%t/execroot \
# RUN:     %{inputs}/test-output-root | \
# RUN:   FileCheck --check-prefix=ROOTED %s -DTESTDIR=%t

# Without the option, the temp path stays under the exec root.
# RUN: rm -rf %t && mkdir -p %t/execroot
# RUN: %{lit} -a -Dexec_root=%t/execroot \
# RUN:     %{inputs}/test-output-root | \
# RUN:   FileCheck --check-prefix=DEFAULT %s -DTESTDIR=%t

# ROOTED: TEMP_PATH=[[TESTDIR]]/out/output-root-suite/{{.*}}Output

# DEFAULT: TEMP_PATH=[[TESTDIR]]/execroot/{{.*}}Output
