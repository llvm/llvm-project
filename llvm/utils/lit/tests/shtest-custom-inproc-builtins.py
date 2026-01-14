## This test provides some custom in-process built-ins via the lit.cfg, and
## verifies that their output can be redirected correctly.
#
# RUN: %{lit} -v %{inputs}/shtest-custom-inproc-builtins \
# RUN: | FileCheck -match-full-lines %s
# END.

# CHECK: PASS: shtest-custom-inproc-builtins :: use-custom-inproc-builtins.txt ({{[^)]*}})

# CHECK: Passed: 1 ({{[^)]*}})
