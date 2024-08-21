## Check that the 'unset' command fails as expected for various tests.

# RUN: not %{lit} -a -v %{inputs}/shtest-unset \
# RUN: | FileCheck -match-full-lines %s
#
# END.

## Check that the 'unset' command's expected failures.

# CHECK: -- Testing: 4 tests{{.*}}

# CHECK: FAIL: shtest-unset :: unset-multiple-variables.txt{{.*}}
# CHECK: unset FOO BAR
# CHECK-NEXT: # executed command: unset FOO BAR
# CHECK-NEXT: # .---command stderr------------
# CHECK-NEXT: # | 'unset' command is not supported by the lit internal shell. Please use 'env -u VARIABLE' to unset environment variables.
# CHECK: # error: command failed with exit status: 127

# CHECK: FAIL: shtest-unset :: unset-no-args.txt{{.*}}
# CHECK: unset
# CHECK-NEXT: # executed command: unset
# CHECK-NEXT: # .---command stderr------------
# CHECK-NEXT: # | 'unset' command is not supported by the lit internal shell. Please use 'env -u VARIABLE' to unset environment variables.
# CHECK: # error: command failed with exit status: 127

# CHECK: FAIL: shtest-unset :: unset-nonexistent-variable.txt{{.*}}
# CHECK: unset NONEXISTENT
# CHECK-NEXT: # executed command: unset NONEXISTENT
# CHECK-NEXT: # .---command stderr------------
# CHECK-NEXT: # | 'unset' command is not supported by the lit internal shell. Please use 'env -u VARIABLE' to unset environment variables.
# CHECK: # error: command failed with exit status: 127

# CHECK: FAIL: shtest-unset :: unset-variable.txt{{.*}}
# CHECK: unset FOO
# CHECK-NEXT: # executed command: unset FOO
# CHECK-NEXT: # .---command stderr------------
# CHECK-NEXT: # | 'unset' command is not supported by the lit internal shell. Please use 'env -u VARIABLE' to unset environment variables.
# CHECK: # error: command failed with exit status: 127

# CHECK: Total Discovered Tests: 4
# CHECK: Failed: 4 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NOT: {{.}}
