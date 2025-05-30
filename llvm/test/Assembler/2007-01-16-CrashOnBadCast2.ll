; PR1117
; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: invalid cast opcode for cast from 'i64' to 'ptr'
@X = constant ptr trunc (i64 0 to ptr)
