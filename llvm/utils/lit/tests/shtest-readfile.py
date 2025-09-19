## Tests the readfile substitution.

# RUN: env LIT_USE_INTERNAL_SHELL=1  not %{lit} -a -v %{inputs}/shtest-readfile | FileCheck -match-full-lines -DTEMP_PATH=%S%{fs-sep}Inputs%{fs-sep}shtest-readfile%{fs-sep}Output %s

# CHECK: -- Testing: 4 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-readfile :: absolute-paths.txt ({{[^)]*}})
# CHECK: echo hello
# CHECK: # executed command: echo '%{readfile:[[TEMP_PATH]]{{[\\\/]}}absolute-paths.txt.tmp}'

# CHECK-LABEL: FAIL: shtest-readfile :: file-does-not-exist.txt ({{[^)]*}})
# CHECK: # executed command: @echo 'echo %{readfile:/file/does/not/exist}'
# CHECK: # | File specified in readfile substitution does not exist: /file/does/not/exist

# CHECK-LABEL: FAIL: shtest-readfile :: relative-paths.txt ({{[^)]*}})
# CHECK: echo hello
# CHECK: # executed command: echo '%{readfile:rel_path_test_folder/test_file}'

# CHECK-LABEL: FAIL: shtest-readfile :: two-same-line.txt ({{[^)]*}})
# CHECK: echo hello bye
# CHECK: # executed command: echo '%{readfile:[[TEMP_PATH]]{{[\\\/]}}two-same-line.txt.tmp.1}' '%{readfile:[[TEMP_PATH]]{{[\\\/]}}two-same-line.txt.tmp.2}'
