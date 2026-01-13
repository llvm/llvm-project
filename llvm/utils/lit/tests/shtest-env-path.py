## Tests env command for setting the PATH variable.

# The test is using /bin/sh. Limit to system known to have /bin/sh.
# REQUIRES: system-linux || system-darwin

# RUN: %{lit} -a %{inputs}/shtest-env-path/path.txt \
# RUN:   | FileCheck -match-full-lines %s
#
# END.

# CHECK: -- Testing: 1 tests{{.*}}
# CHECK: PASS: shtest-env-path :: path.txt (1 of 1)
# CHECK: --
