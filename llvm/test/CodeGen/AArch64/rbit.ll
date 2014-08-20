; RUN: llc -mtriple=aarch64-eabi %s -o - | FileCheck %s

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
