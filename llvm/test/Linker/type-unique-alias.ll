; RUN: llvm-link -S %s %p/Inputs/type-unique-alias.ll | FileCheck %s

%t = type { i8 }

@g = global %t zeroinitializer
@a = weak alias %t, ptr @g

; CHECK: @g = global %t zeroinitializer
; CHECK: @g2 = global %t zeroinitializer
; CHECK: @a = weak alias %t, ptr @g
