; RUN: llvm-link %s %S/Inputs/ConstantGlobals.ll -S | FileCheck %s
; RUN: llvm-link %S/Inputs/ConstantGlobals.ll %s -S | FileCheck %s

; CHECK-DAG: @X = constant [1 x i32] [i32 8]
@X = external global [1 x i32]

; CHECK-DAG: @Y = external global [1 x i32]
@Y = external global [1 x i32]

define [1 x i32]* @use-Y() {
  ret [1 x i32] *@Y
}
