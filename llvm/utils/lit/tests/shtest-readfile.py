## Tests the readfile substitution

# RUN: not %{lit} -a -v %{inputs}/shtest-readfile | FileCheck -match-full-lines %s

# CHECK: -- Testing: 3 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-readfile :: absolute-paths.txt ({{[^)]*}})
# CHECK: echo hello
# CHECK: # executed command: echo '%{readfile:{{.*}}}'

# CHECK-LABEL: FAIL: shtest-readfile :: relative-paths.txt ({{[^)]*}})
# CHECK: echo hello
# CHECK: # executed command: echo '%{readfile:rel_path_test_folder/test_file}'

# CHECK-LABEL: FAIL: shtest-readfile :: two-same-line.txt ({{[^)]*}})
# CHECK: echo hello bye
# CHECK: # executed command: echo '%{readfile:{{.*}}.1}' '%{readfile:{{.*}}.2}'
