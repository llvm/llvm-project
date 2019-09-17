# Check the env command
#
# RUN: not %{lit} -j 1 -a -v %{inputs}/shtest-env \
# RUN: | FileCheck -match-full-lines %s
#
# END.

# Make sure env commands are included in printed commands.

# CHECK: -- Testing: 7 tests{{.*}}

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

# CHECK: PASS: shtest-env :: env-u.txt ({{[^)]*}})
# CHECK: $ "{{[^"]*}}" "print_environment.py"
# CHECK: $ "env" "-u" "FOO" "{{[^"]*}}" "print_environment.py"
# CHECK: $ "env" "-u" "FOO" "-u" "BAR" "{{[^"]*}}" "print_environment.py"

# CHECK: PASS: shtest-env :: env.txt ({{[^)]*}})
# CHECK: $ "env" "A_FOO=999" "{{[^"]*}}" "print_environment.py"
# CHECK: $ "env" "A_FOO=1" "B_BAR=2" "C_OOF=3" "{{[^"]*}}" "print_environment.py"

# CHECK: PASS: shtest-env :: mixed.txt ({{[^)]*}})
# CHECK: $ "env" "A_FOO=999" "-u" "FOO" "{{[^"]*}}" "print_environment.py"
# CHECK: $ "env" "A_FOO=1" "-u" "FOO" "B_BAR=2" "-u" "BAR" "C_OOF=3" "{{[^"]*}}" "print_environment.py"

# CHECK: Expected Passes : 3
# CHECK: Unexpected Failures: 4
# CHECK-NOT: {{.}}
