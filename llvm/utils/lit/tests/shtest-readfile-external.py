## Tests the readfile substitution.

# TODO(boomanaiden154): This sometimes fails, possibly due to buffers not being flushed.
# ALLOW_RETRIES: 2

# UNSUPPORTED: system-windows
# RUN: env LIT_USE_INTERNAL_SHELL=0 not %{lit} -v %{inputs}/shtest-readfile | FileCheck -match-full-lines -DTEMP_PATH=%S/Inputs/shtest-readfile/Output %s

# CHECK: -- Testing: 5 tests{{.*}}

# CHECK-LABEL: FAIL: shtest-readfile :: absolute-paths.txt ({{[^)]*}})
# CHECK: echo $(cat [[TEMP_PATH]]/absolute-paths.txt.tmp) && test -e [[TEMP_PATH]]/absolute-paths.txt.tmp {{.*}}
# CHECK: + echo hello

# CHECK-LABEL: FAIL: shtest-readfile :: file-does-not-exist.txt ({{[^)]*}})
# CHECK: echo $(cat /file/does/not/exist) && test -e /file/does/not/exist {{.*}}
# CHECK: {{.*}}cat{{.*}}/file/does/not/exist{{.*}}

# CHECK-LABEL: FAIL: shtest-readfile :: relative-paths.txt ({{[^)]*}})
# CHECK: echo $(cat rel_path_test_folder/test_file) && test -e rel_path_test_folder/test_file {{.*}}
# CHECK: + echo hello

# CHECK-LABEL: FAIL: shtest-readfile :: two-same-line.txt ({{[^)]*}})
# CHECK: echo $(cat [[TEMP_PATH]]/two-same-line.txt.tmp.1) $(cat [[TEMP_PATH]]/two-same-line.txt.tmp.2) && test -e [[TEMP_PATH]]/two-same-line.txt.tmp.1 && test -e [[TEMP_PATH]]/two-same-line.txt.tmp.2 {{.*}}
# CHECK: + echo hello bye
