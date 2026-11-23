; Found by inspection of the code
; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: initializer with struct type has wrong # elements

@0 = global {} { i32 7, float 1.0, i32 7, i32 8 }
