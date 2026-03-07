# Check the json-diff builtin command

# RUN: %{lit} -a %{inputs}/shtest-json-diff | FileCheck -match-full-lines %s
# END.

# CHECK: -- Testing: 13 tests{{.*}}

# CHECK: PASS: shtest-json-diff :: context.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: different.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: duplicate-keys.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: identical.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: ignore-extra-keys.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: invalid-input.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: missing-file.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: patterns-absence.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: patterns-capture.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: patterns-combined.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: patterns-escape.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: patterns-regex.txt {{.*}}
# CHECK: PASS: shtest-json-diff :: patterns-template.txt {{.*}}

# CHECK:  Passed: 13{{.*}}
