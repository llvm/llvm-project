## Test the export command.

# RUN: not %{lit} -a -v %{inputs}/shtest-export \
# RUN: | FileCheck -match-full-lines %s
#
# END.

# CHECK: FAIL: shtest-export :: export-too-many-args.txt {{.*}}
# CHECK: export FOO=1 BAR=2
# CHECK: # executed command: export FOO=1 BAR=2
# CHECK: # | 'export' supports only one argument
# CHECK: # error: command failed with exit status: {{.*}}
