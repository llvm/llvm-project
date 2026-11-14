## Test the echo command.

# RUN: %{lit} -v %{inputs}/shtest-echo \
# RUN: | FileCheck -match-full-lines %s
# END.

# CHECK: PASS: shtest-echo :: echo-escape.txt ({{[^)]*}})
# CHECK: PASS: shtest-echo :: echo.txt ({{[^)]*}})

# CHECK: Total Discovered Tests: 2
# CHECK-NEXT: Passed: 2 (100.00%)
