; RUN: echo "%X = type { i32 } @G2 = global %X { i32 4 }" > %t.ll
; RUN: llvm-link %s %t.ll -S | FileCheck %s
; PR11464

%X = type { i32 }
@G = global %X { i32 4 }


; CHECK: @G = global %X { i32 4 }
; CHECK: @G2 = global %X { i32 4 }
