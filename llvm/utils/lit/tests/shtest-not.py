# Check the not command
#
# RUN: not %{lit} -j 1 -a -v %{inputs}/shtest-not \
# RUN: | FileCheck -match-full-lines %s
#
# END.

# Make sure not and env commands are included in printed commands.

# CHECK: -- Testing: 13 tests{{.*}}

# CHECK: FAIL: shtest-not :: not-args-last-is-crash.txt {{.*}}
# CHECK: $ "not" "--crash"
# CHECK: Error: 'not' requires a subcommand
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-not :: not-args-nested-none.txt {{.*}}
# CHECK: $ "not" "not" "not"
# CHECK: Error: 'not' requires a subcommand
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-not :: not-args-none.txt {{.*}}
# CHECK: $ "not"
# CHECK: Error: 'not' requires a subcommand
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-not :: not-calls-cd.txt {{.*}}
# CHECK: $ "not" "not" "cd" "foobar"
# CHECK: $ "not" "--crash" "cd" "foobar"
# CHECK: Error: 'not --crash' cannot call 'cd'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-not :: not-calls-colon.txt {{.*}}
# CHECK: $ "not" "not" ":" "foobar"
# CHECK: $ "not" "--crash" ":"
# CHECK: Error: 'not --crash' cannot call ':'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-not :: not-calls-diff-with-crash.txt {{.*}}
# CHECK: $ "not" "--crash" "diff" "-u" {{.*}}
# CHECK-NOT: "$"
# CHECK-NOT: {{[Ee]rror}}
# CHECK: error: command failed with exit status: {{.*}}
# CHECK-NOT: {{[Ee]rror}}
# CHECK-NOT: "$"

# CHECK: FAIL: shtest-not :: not-calls-diff.txt {{.*}}
# CHECK: $ "not" "diff" {{.*}}
# CHECK: $ "not" "not" "not" "diff" {{.*}}
# CHECK: $ "not" "not" "not" "not" "not" "diff" {{.*}}
# CHECK: $ "diff" {{.*}}
# CHECK: $ "not" "not" "diff" {{.*}}
# CHECK: $ "not" "not" "not" "not" "diff" {{.*}}
# CHECK: $ "not" "diff" {{.*}}
# CHECK-NOT: "$"

# CHECK: FAIL: shtest-not :: not-calls-echo.txt {{.*}}
# CHECK: $ "not" "not" "echo" "hello" "world"
# CHECK: $ "not" "--crash" "echo" "hello" "world"
# CHECK: Error: 'not --crash' cannot call 'echo'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-not :: not-calls-env-builtin.txt {{.*}}
# CHECK: $ "not" "--crash" "env" "-u" "FOO" "BAR=3" "rm" "{{.*}}.no-such-file"
# CHECK: Error: 'env' cannot call 'rm'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-not :: not-calls-export.txt {{.*}}
# CHECK: $ "not" "not" "export" "FOO=1"
# CHECK: $ "not" "--crash" "export" "BAZ=3"
# CHECK: Error: 'not --crash' cannot call 'export'
# CHECK: error: command failed with exit status: {{.*}}


# CHECK: PASS: shtest-not :: not-calls-external.txt {{.*}}

# CHECK: $ "not" "{{[^"]*}}" "fail.py"
# CHECK: $ "not" "not" "{{[^"]*}}" "pass.py"
# CHECK: $ "not" "not" "not" "{{[^"]*}}" "fail.py"
# CHECK: $ "not" "not" "not" "not" "{{[^"]*}}" "pass.py"

# CHECK: $ "not" "not" "--crash" "{{[^"]*}}" "pass.py"
# CHECK: $ "not" "not" "--crash" "{{[^"]*}}" "fail.py"
# CHECK: $ "not" "not" "--crash" "not" "{{[^"]*}}" "pass.py"
# CHECK: $ "not" "not" "--crash" "not" "{{[^"]*}}" "fail.py"

# CHECK: $ "env" "not" "{{[^"]*}}" "fail.py"
# CHECK: $ "not" "env" "{{[^"]*}}" "fail.py"
# CHECK: $ "env" "FOO=1" "not" "{{[^"]*}}" "fail.py"
# CHECK: $ "not" "env" "FOO=1" "BAR=1" "{{[^"]*}}" "fail.py"
# CHECK: $ "env" "FOO=1" "BAR=1" "not" "env" "-u" "FOO" "BAR=2" "{{[^"]*}}" "fail.py"
# CHECK: $ "not" "env" "FOO=1" "BAR=1" "not" "env" "-u" "FOO" "-u" "BAR" "{{[^"]*}}" "pass.py"
# CHECK: $ "not" "not" "env" "FOO=1" "env" "FOO=2" "BAR=1" "{{[^"]*}}" "pass.py"
# CHECK: $ "env" "FOO=1" "-u" "BAR" "env" "-u" "FOO" "BAR=1" "not" "not" "{{[^"]*}}" "pass.py"

# CHECK: $ "not" "env" "FOO=1" "BAR=1" "env" "FOO=2" "BAR=2" "not" "--crash" "{{[^"]*}}" "pass.py"
# CHECK: $ "not" "env" "FOO=1" "BAR=1" "not" "--crash" "not" "{{[^"]*}}" "pass.py"
# CHECK: $ "not" "not" "--crash" "env" "-u" "BAR" "not" "env" "-u" "FOO" "BAR=1" "{{[^"]*}}" "pass.py"


# CHECK: FAIL: shtest-not :: not-calls-mkdir.txt {{.*}}
# CHECK: $ "not" "mkdir" {{.*}}
# CHECK: $ "not" "--crash" "mkdir" "foobar"
# CHECK: Error: 'not --crash' cannot call 'mkdir'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-not :: not-calls-rm.txt {{.*}}
# CHECK: $ "not" "rm" {{.*}}
# CHECK: $ "not" "--crash" "rm" "foobar"
# CHECK: Error: 'not --crash' cannot call 'rm'
# CHECK: error: command failed with exit status: {{.*}}

# CHECK: Passed:  1
# CHECK: Failed: 12
# CHECK-NOT: {{.}}
