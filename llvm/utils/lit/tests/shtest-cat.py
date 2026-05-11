## Test the cat command.

## This is required for the use of %errc_ENOENT.
# REQUIRES: llvm-config-available
#
# RUN: not %{lit} -v %{inputs}/shtest-cat \
# RUN: | FileCheck -match-full-lines -DERROR_MSG=%errc_ENOENT %s
# END.

# CHECK: FAIL: shtest-cat :: cat-error-0.txt ({{[^)]*}})
# CHECK: cat -b temp1.txt
# CHECK: # .---command stderr{{-*}}
# CHECK-NEXT: # | Unsupported: 'cat':  option -b not recognized
# CHECK: # error: command failed with exit status: 1

# CHECK: FAIL: shtest-cat :: cat-error-1.txt ({{[^)]*}})
# CHECK: cat temp1.txt
# CHECK: # .---command stderr{{-*}}
# CHECK-NEXT: # | [Errno {{[0-9]+}}] [[ERROR_MSG]]: 'temp1.txt'
# CHECK: # error: command failed with exit status: 1

# CHECK: PASS: shtest-cat :: cat.txt ({{[^)]*}})

# CHECK: Total Discovered Tests: 3
# CHECK-NEXT: Passed: 1 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NEXT: Failed: 2 {{\([0-9]*\.[0-9]*%\)}}
