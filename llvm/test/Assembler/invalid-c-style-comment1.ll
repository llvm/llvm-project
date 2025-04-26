; RUN: not llvm-as --disable-output %s 2>&1 | FileCheck %s -DFILE=%s

@B = external global i32

/*   /* Nested comments not supported */

; CHECK: [[FILE]]:[[@LINE+1]]:1: error: redefinition of global '@B'
@B = external global i32
