# Check the various features of the ShTest format.
#
# RUN: not %{lit} -j 1 -v %{inputs}/shtest-output-printing > %t.out
# RUN: FileCheck < %t.out %s
#
# END.

# CHECK: -- Testing:

# CHECK: FAIL: shtest-output-printing :: basic.txt
# CHECK-NEXT: *** TEST 'shtest-output-printing :: basic.txt' FAILED ***
# CHECK-NEXT: Script:
# CHECK-NEXT: --
# CHECK:      --
# CHECK-NEXT: Exit Code: 1
#
# CHECK:      Command Output
# CHECK-NEXT: --
# CHECK-NEXT: $ "true"
# CHECK-NEXT: $ "echo" "hi"
# CHECK-NEXT: # command output:
# CHECK-NEXT: hi
#
# CHECK:      $ "false"
# CHECK-NEXT: note: command had no output on stdout or stderr
# CHECK-NEXT: error: command failed with exit status: 1
