## Tests glob pattern handling in echo command.

# RUN: not %{lit} -v %{inputs}/shtest-glob \
# RUN: | FileCheck -dump-input=fail -match-full-lines --implicit-check-not=Error: %s
# END.

# CHECK: UNRESOLVED: shtest-glob :: glob-echo.txt ({{[^)]*}})
# CHECK: TypeError: string argument expected, got 'GlobItem'

# CHECK:      FAIL: shtest-glob :: glob-mkdir.txt ({{[^)]*}})
# CHECK:      # | Error: 'mkdir' command failed, {{.*}}example_file1.input'
# CHECK-NEXT: # | Error: 'mkdir' command failed, {{.*}}example_file2.input'
# CHECK:      # error: command failed with exit status: 1
