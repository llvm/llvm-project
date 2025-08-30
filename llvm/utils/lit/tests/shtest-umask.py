# Check the umask command

# RUN: not %{lit} -a -v %{inputs}/shtest-umask | FileCheck -match-full-lines %s
# TODO(boomanaiden154): We should be asserting that we get expected behavior
# on Windows rather than just listing this as unsupported.
# UNSUPPORTED: system-windows

# CHECK: -- Testing: 3 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-umask :: umask-bad-arg.txt ({{[^)]*}})
# CHECK: umask bad
# CHECK: # | Error: 'umask': invalid literal {{.*}}

# CHECK-LABEL: FAIL: shtest-umask :: umask-too-many-args.txt ({{[^)]*}})
# CHECK: umask 0 0
# CHECK: # | 'umask' supports only one argument

# CHECK: Total Discovered Tests: 3
# CHECK: {{Passed|Unsupported}}: 1 {{\([0-9]*\.[0-9]*%\)}}
# CHECK: Failed{{ *}}: 2 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NOT: {{.}}
