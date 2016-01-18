; RUN: llvm-link %p/opaque.ll %p/Inputs/opaque.ll -S -o - | FileCheck %s

; CHECK-DAG: %A =   type {}
; CHECK-DAG: %B =   type { %C, %C, %B* }
; CHECK-DAG: %B.1 = type { %D, %E, %B.1* }
; CHECK-DAG: %C =   type { %A }
; CHECK-DAG: %D =   type { %E }
; CHECK-DAG: %E =   type opaque

; CHECK-DAG: @g1 = external global %B
; CHECK-DAG: @g2 = external global %A
; CHECK-DAG: @g3 = external global %B.1

; CHECK-DAG: getelementptr %A, %A* null, i32 0

%A = type opaque
%B = type { %C, %C, %B* }

%C = type { %A }

@g1 = external global %B
