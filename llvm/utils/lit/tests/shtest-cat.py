## Test the cat command.
#
# RUN: not %{lit} -a -v %{inputs}/shtest-cat \
# RUN: | FileCheck -match-full-lines %s
# END.

# CHECK: FAIL: shtest-cat :: cat-error-0.txt ({{[^)]*}})
# CHECK: cat -b temp1.txt
# CHECK: # .---command stderr{{-*}}
# CHECK-NEXT: # | Unsupported: 'cat':  option -b not recognized
# CHECK: # error: command failed with exit status: 1

# CHECK: FAIL: shtest-cat :: cat-error-1.txt ({{[^)]*}})
# CHECK: cat temp1.txt
# CHECK: # .---command stderr{{-*}}
# CHECK-NEXT: # | [Errno 2] No such file or directory: 'temp1.txt'
# CHECK: # error: command failed with exit status: 1

# CHECK: PASS: shtest-cat :: cat.txt ({{[^)]*}})

# CHECK: Total Discovered Tests: 3
# CHECK-NEXT: Passed: 1 {{\([0-9]*\.[0-9]*%\)}}
# CHECK-NEXT: Failed: 2 {{\([0-9]*\.[0-9]*%\)}}
