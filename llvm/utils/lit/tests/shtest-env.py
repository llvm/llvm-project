# Check the env command
#
# RUN: %{lit} -j 1 -a -v %{inputs}/shtest-env \
# RUN: | FileCheck -match-full-lines %s
#
# END.

# Make sure env commands are included in printed commands.

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
