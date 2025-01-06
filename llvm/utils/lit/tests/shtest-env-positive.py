## Test the env command (passing tests).

# RUN: %{lit} -a -v %{inputs}/shtest-env-positive \
# RUN: | FileCheck -match-full-lines %s
#
# END.

## Test the env command's successful executions.

# CHECK: -- Testing: 9 tests{{.*}}

# CHECK: PASS: shtest-env :: env-args-last-is-assign.txt ({{[^)]*}})
# CHECK: env FOO=1
# CHECK: # executed command: env FOO=1
# CHECK-NOT: # error:
# CHECK: --

# CHECK: PASS: shtest-env :: env-args-last-is-u-arg.txt ({{[^)]*}})
# CHECK: env -u FOO
# CHECK: # executed command: env -u FOO
# CHECK-NOT: # error:
# CHECK: --

# CHECK: PASS: shtest-env :: env-args-last-is-u.txt ({{[^)]*}})
# CHECK: env -u
# CHECK: # executed command: env -u
# CHECK-NOT: # error:
# CHECK: --

# CHECK: PASS: shtest-env :: env-args-nested-none.txt ({{[^)]*}})
# CHECK: env env env
# CHECK: # executed command: env env env
# CHECK-NOT: # error:
# CHECK: --

# CHECK: PASS: shtest-env :: env-calls-env.txt ({{[^)]*}})
# CHECK: env env | {{.*}}
# CHECK: # executed command: env env
# CHECK-NOT: # error:
# CHECK: --

# CHECK: PASS: shtest-env :: env-no-subcommand.txt ({{[^)]*}})
# CHECK: env | {{.*}}
# CHECK: # executed command: env
# CHECK: env FOO=2 BAR=1 | {{.*}}
# CHECK: # executed command: env FOO=2 BAR=1
# CHECK-NOT: # error:
# CHECK: --

# CHECK: PASS: shtest-env :: env-u.txt ({{[^)]*}})
# CHECK: env -u FOO | {{.*}}
# CHECK: # executed command: env -u FOO
# CHECK-NOT: # error:
# CHECK: --

# CHECK: PASS: shtest-env :: env.txt ({{[^)]*}})
# CHECK: env A_FOO=999 | {{.*}}
# CHECK: # executed command: env A_FOO=999
# CHECK-NOT: # error:
# CHECK: --

# CHECK: PASS: shtest-env :: mixed.txt ({{[^)]*}})
# CHECK: env A_FOO=999 -u FOO | {{.*}}
# CHECK: # executed command: env A_FOO=999 -u FOO
# CHECK-NOT: # error:
# CHECK: --

# CHECK: Total Discovered Tests: 9
# CHECK: Passed: 9 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NOT: {{.}}
