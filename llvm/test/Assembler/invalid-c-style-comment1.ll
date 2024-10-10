; RUN: not llvm-as --disable-output %s 2>&1 | FileCheck %s -DFILE=%s

@B = external global i32

; CHECK: [[FILE]]:[[@LINE+1]]:1: error: Unterminated comment!
/* End of the assembly file
   /* Unterminated comment with multiple nesting depths */
   /* /* ignored */ */
   /* /* /* ignored */ */ */
* /   

