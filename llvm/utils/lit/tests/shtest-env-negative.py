## Test the env command (failing tests).

# RUN: not %{lit} -a -v %{inputs}/shtest-env-negative \
# RUN: | FileCheck -match-full-lines %s
#
# END.

## Test the env command's expected failures.

# CHECK: -- Testing: 7 tests{{.*}}

# CHECK: FAIL: shtest-env :: env-calls-cd.txt ({{[^)]*}})
# CHECK: env -u FOO BAR=3 cd foobar
# CHECK: # executed command: env -u FOO BAR=3 cd foobar
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-colon.txt ({{[^)]*}})
# CHECK: env -u FOO BAR=3 :
# CHECK: # executed command: env -u FOO BAR=3 :
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-echo.txt ({{[^)]*}})
# CHECK: env -u FOO BAR=3 echo hello world
# CHECK: # executed command: env -u FOO BAR=3 echo hello world
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-export.txt ({{[^)]*}})
# CHECK: env -u FOO BAR=3 export BAZ=3
# CHECK: # executed command: env -u FOO BAR=3 export BAZ=3
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-mkdir.txt ({{[^)]*}})
# CHECK: env -u FOO BAR=3 mkdir foobar
# CHECK: # executed command: env -u FOO BAR=3 mkdir foobar
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-not-builtin.txt ({{[^)]*}})
# CHECK: env -u FOO BAR=3 not rm {{.+}}.no-such-file
# CHECK: # executed command: env -u FOO BAR=3 not rm {{.+}}.no-such-file{{.*}}
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-env :: env-calls-rm.txt ({{[^)]*}})
# CHECK: env -u FOO BAR=3 rm foobar
# CHECK: # executed command: env -u FOO BAR=3 rm foobar
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: Total Discovered Tests: 7
# CHECK: Failed: 7 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NOT: {{.}}
