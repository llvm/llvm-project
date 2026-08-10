; RUN: llc -mtriple=aarch64 %s -o - | FileCheck %s

; The llvm.aarch64.rbit intrinsic should be auto-upgraded to the
; target-independent bitreverse intrinsic.

; CHECK-LABEL: rbit32
; CHECK: rbit w0, w0
define i32 @rbit32(i32 %t) {
entry:
  %rbit.i = call i32 @llvm.aarch64.rbit.i32(i32 %t)
  ret i32 %rbit.i
}

; CHECK-LABEL: rbit64
; CHECK: rbit x0, x0
define i64 @rbit64(i64 %t) {
entry:
  %rbit.i = call i64 @llvm.aarch64.rbit.i64(i64 %t)
  ret i64 %rbit.i
}

declare i64 @llvm.aarch64.rbit.i64(i64)
declare i32 @llvm.aarch64.rbit.i32(i32)

; CHECK-LABEL: rbit_generic32
; CHECK: rbit w0, w0
define i32 @rbit_generic32(i32 %t) {
entry:
  %rbit = call i32 @llvm.bitreverse.i32(i32 %t)
  ret i32 %rbit
}

; CHECK-LABEL: rbit_generic64
; CHECK: rbit x0, x0
define i64 @rbit_generic64(i64 %t) {
entry:
  %rbit = call i64 @llvm.bitreverse.i64(i64 %t)
  ret i64 %rbit
}

declare i32 @llvm.bitreverse.i32(i32) readnone
declare i64 @llvm.bitreverse.i64(i64) readnone
