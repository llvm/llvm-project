; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@GV = external global token
; CHECK: invalid type for global variable
