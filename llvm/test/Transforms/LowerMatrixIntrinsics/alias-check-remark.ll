; RUN: opt -passes=lower-matrix-intrinsics -aa-pipeline=basic-aa -pass-remarks-missed=lower-matrix-intrinsics %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Adding run-time alias check between load operand #1 of a matrix multiply and the store of the result
; CHECK-NOT: Adding run-time alias check between load operand

define void @multiply(ptr noalias %A, ptr %B, ptr %C) {
entry:
  %a = load <64 x double>, ptr %A, align 16
  %b = load <64 x double>, ptr %B, align 16

  %c = call <64 x double> @llvm.matrix.multiply(<64 x double> %a, <64 x double> %b, i32 8, i32 8, i32 8)

  store <64 x double> %c, ptr %C, align 16
  ret void
}

declare <64 x double> @llvm.matrix.multiply(<64 x double>, <64 x double>, i32, i32, i32)
