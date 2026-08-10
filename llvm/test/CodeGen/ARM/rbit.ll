; RUN: llc -mtriple=armv8-eabi %s -o - | FileCheck %s

; CHECK-LABEL: rbit
; CHECK: rbit r0, r0
define i32 @rbit(i32 %t) {
entry:
  %rbit = call i32 @llvm.arm.rbit(i32 %t)
  ret i32 %rbit
}

; CHECK-LABEL: rbit_constant
; CHECK: mov r0, #0
; CHECK-NOT: rbit
; CHECK: bx lr
define i32 @rbit_constant() {
entry:
  %rbit.i = call i32 @llvm.arm.rbit(i32 0)
  ret i32 %rbit.i
}

declare i32 @llvm.arm.rbit(i32)

declare i32 @llvm.bitreverse.i32(i32) readnone

; CHECK-LABEL: rbit_generic
; CHECK: rbit r0, r0
define i32 @rbit_generic(i32 %t) {
entry:
  %rbit = call i32 @llvm.bitreverse.i32(i32 %t)
  ret i32 %rbit
}

