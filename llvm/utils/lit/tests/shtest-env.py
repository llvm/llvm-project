# Check the env command
#
# RUN: not %{lit} -j 1 -a -v %{inputs}/shtest-env \
# RUN: | FileCheck -match-full-lines %s
#
# END.

# Make sure env commands are included in printed commands.

# CHECK: -- Testing: 13 tests{{.*}}

# CHECK: FAIL: shtest-env :: env-args-last-is-assign.txt ({{[^)]*}})
# CHECK: Error: 'env' requires a subcommand
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-args-last-is-u-arg.txt ({{[^)]*}})
# CHECK: Error: 'env' requires a subcommand
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-args-last-is-u.txt ({{[^)]*}})
# CHECK: Error: 'env' requires a subcommand
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-args-none.txt ({{[^)]*}})
# CHECK: Error: 'env' requires a subcommand
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-cd.txt ({{[^)]*}})
# CHECK: $ "env" "-u" "FOO" "BAR=3" "cd" "foobar"
# CHECK: Error: 'env' cannot call 'cd'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-colon.txt ({{[^)]*}})
# CHECK: $ "env" "-u" "FOO" "BAR=3" ":"
# CHECK: Error: 'env' cannot call ':'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-echo.txt ({{[^)]*}})
# CHECK: $ "env" "-u" "FOO" "BAR=3" "echo" "hello" "world"
# CHECK: Error: 'env' cannot call 'echo'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-export.txt ({{[^)]*}})
# CHECK: $ "env" "-u" "FOO" "BAR=3" "export" "BAZ=3"
# CHECK: Error: 'env' cannot call 'export'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-mkdir.txt ({{[^)]*}})
# CHECK: $ "env" "-u" "FOO" "BAR=3" "mkdir" "foobar"
# CHECK: Error: 'env' cannot call 'mkdir'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-rm.txt ({{[^)]*}})
# CHECK: $ "env" "-u" "FOO" "BAR=3" "rm" "foobar"
# CHECK: Error: 'env' cannot call 'rm'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: PASS: shtest-env :: env-u.txt ({{[^)]*}})
# CHECK: $ "{{[^"]*}}" "print_environment.py"
# CHECK: $ "env" "-u" "FOO" "{{[^"]*}}" "print_environment.py"
# CHECK: $ "env" "-u" "FOO" "-u" "BAR" "{{[^"]*}}" "print_environment.py"
# CHECK-NOT: ${{.*}}print_environment.py

# CHECK: PASS: shtest-env :: env.txt ({{[^)]*}})
# CHECK: $ "env" "A_FOO=999" "{{[^"]*}}" "print_environment.py"
# CHECK: $ "env" "A_FOO=1" "B_BAR=2" "C_OOF=3" "{{[^"]*}}" "print_environment.py"
# CHECK-NOT: ${{.*}}print_environment.py

# CHECK: PASS: shtest-env :: mixed.txt ({{[^)]*}})
# CHECK: $ "env" "A_FOO=999" "-u" "FOO" "{{[^"]*}}" "print_environment.py"
# CHECK: $ "env" "A_FOO=1" "-u" "FOO" "B_BAR=2" "-u" "BAR" "C_OOF=3" "{{[^"]*}}" "print_environment.py"
# CHECK-NOT: ${{.*}}print_environment.py

# CHECK: Expected Passes : 3
# CHECK: Unexpected Failures: 10
# CHECK-NOT: {{.}}
