; RUN: llvm-link %p/opaque.ll %p/Inputs/opaque.ll -S -o - | FileCheck %s

; CHECK-DAG: %A =   type {}
; CHECK-DAG: %B =   type { %C, %C, ptr }
; CHECK-DAG: %B.1 = type { %D, %E, ptr }
; CHECK-DAG: %C =   type { %A }
; CHECK-DAG: %D =   type { %E }
; CHECK-DAG: %E =   type opaque

; CHECK-DAG: @g1 = external global %B
; CHECK-DAG: @g2 = external global %A
; CHECK-DAG: @g3 = external global %B.1

; CHECK-DAG: getelementptr %A, ptr null, i32 0

%A = type opaque
%B = type { %C, %C, ptr }

%C = type { %A }

@g1 = external global %B

define ptr @use_g1() {
  ret ptr @g1
}
