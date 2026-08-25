# Check that --test-output-root relocates each suite's writable output tree to
# <root>/<suite-name>, so %t/Output/etc. resolve under the given root directory
# instead of the suite's build test_exec_root.
#
# Every case captures the outer lit's %t as TESTDIR and matches the inner temp
# path against it exactly, so each check fully pins where the output landed.
#
# TESTDIR uses %/t rather than %t: %{lit} runs the inner lit with normalized
# slashes, so the temp path it reports is always forward-slashed, while the
# outer %t is native (backslashed on Windows).

# With the option, the temp path is under <root>/<suite-name>, not the exec root.
# RUN: rm -rf %t && mkdir -p %t/execroot %t/out
# RUN: %{lit} -a --test-output-root %t/out -Dexec_root=%t/execroot \
# RUN:     %{inputs}/test-output-root | \
# RUN:   FileCheck --check-prefix=ROOTED %s -DTESTDIR=%/t

# Without the option, the temp path stays under the exec root.
# RUN: rm -rf %t && mkdir -p %t/execroot
# RUN: %{lit} -a -Dexec_root=%t/execroot \
# RUN:     %{inputs}/test-output-root | \
# RUN:   FileCheck --check-prefix=DEFAULT %s -DTESTDIR=%/t

# A site config defers to the main config, so TestingConfig.finish() runs twice
# and config.name is only set partway through. Check the suite still lands under
# <root>/<suite-name> rather than under a placeholder name.
# RUN: rm -rf %t && mkdir -p %t/execroot %t/out
# RUN: %{lit} -a --test-output-root %t/out -Dexec_root=%t/execroot \
# RUN:     %{inputs}/test-output-root-site/obj | \
# RUN:   FileCheck --check-prefix=SITE-ROOTED %s -DTESTDIR=%/t

# Without the option, the site config's exec root is still honored.
# RUN: rm -rf %t && mkdir -p %t/execroot
# RUN: %{lit} -a -Dexec_root=%t/execroot \
# RUN:     %{inputs}/test-output-root-site/obj | \
# RUN:   FileCheck --check-prefix=SITE-DEFAULT %s -DTESTDIR=%/t

# ROOTED: TEMP_PATH=[[TESTDIR]]{{[\\/]}}out{{[\\/]}}output-root-suite{{[\\/]}}{{.*}}Output

# DEFAULT: TEMP_PATH=[[TESTDIR]]{{[\\/]}}execroot{{[\\/]}}{{.*}}Output

# SITE-ROOTED: TEMP_PATH=[[TESTDIR]]{{[\\/]}}out{{[\\/]}}output-root-site-suite{{[\\/]}}{{.*}}Output

# SITE-DEFAULT: TEMP_PATH=[[TESTDIR]]{{[\\/]}}execroot{{[\\/]}}{{.*}}Output
