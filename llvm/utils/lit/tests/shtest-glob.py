## Tests glob pattern handling in echo command.

# RUN: not %{lit} -a -v %{inputs}/shtest-glob \
# RUN: | FileCheck -dump-input=fail -match-full-lines %s
#
# END.

# CHECK: UNRESOLVED: shtest-glob :: glob-echo.txt ({{[^)]*}})
# CHECK: TypeError: string argument expected, got 'GlobItem'

# CHECK: FAIL: shtest-glob :: glob-mkdir.txt ({{[^)]*}})
# CHECK: # error: command failed with exit status: 1
