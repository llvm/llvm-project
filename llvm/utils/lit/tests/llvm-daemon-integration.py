## Test the integration with LLVM daemon tools.
#
# REQUIRES: have-llvm-build
# RUN: not %{lit} -v %{inputs}/llvm-daemon-integration \
# RUN: | FileCheck --match-full-lines %s
# END.

# CHECK: FAIL: llvm-daemon-integration :: disabled-folder/try-using-exampledaemon.txt ({{.*}})
# CHECK-NEXT: {{\*+}} TEST 'llvm-daemon-integration :: disabled-folder/try-using-exampledaemon.txt' FAILED {{\*+}}
# CHECK-NEXT: Exit Code: 127
# CHECK-EMPTY:
# CHECK-NEXT: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: # RUN: at line {{.*}}
# CHECK-NEXT: %invoke-ExampleDaemon
# CHECK-NEXT: # executed command: %invoke-ExampleDaemon
# CHECK-NEXT: # .---command stderr------------
# CHECK-NEXT: # | '%invoke-ExampleDaemon': command not found
# CHECK-NEXT: # `-----------------------------
# CHECK-NEXT: # error: command failed with exit status: 127

# CHECK: FAIL: llvm-daemon-integration :: disabled-test.txt ({{.*}})
# CHECK-NEXT: {{\*+}} TEST 'llvm-daemon-integration :: disabled-test.txt' FAILED {{\*+}}
# CHECK-NEXT: Exit Code: 127
# CHECK-EMPTY:
# CHECK-NEXT: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: # RUN: at line {{.*}}
# CHECK-NEXT: %invoke-ExampleDaemon
# CHECK-NEXT: # executed command: %invoke-ExampleDaemon
# CHECK-NEXT: # .---command stderr------------
# CHECK-NEXT: # | '%invoke-ExampleDaemon': command not found
# CHECK-NEXT: # `-----------------------------
# CHECK-NEXT: # error: command failed with exit status: 127
# CHECK: PASS: llvm-daemon-integration :: use-daemon.txt ({{.*}})
