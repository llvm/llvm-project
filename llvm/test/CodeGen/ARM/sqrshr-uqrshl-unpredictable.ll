; RUN: llc -mtriple armv8.1m.main -mattr=+mve %s -o - | FileCheck %s

; Check that we don't create an unpredictable sqrshr or uqrshl instruction,
; e.g. sqrshr r0, r0

declare i32 @llvm.arm.mve.sqrshr(i32, i32) #1
declare i32 @llvm.arm.mve.uqrshl(i32, i32) #1

define i32 @sqrshr() #0 {
; CHECK-LABEL: sqrshr
; CHECK-NOT: sqrshr  r[[REG:[0-9]+]], r[[REG]]
  %1 = tail call i32 @llvm.arm.mve.sqrshr(i32 1, i32 1)
  ret i32 %1
}

define i32 @uqrshl() #0 {
; CHECK-LABEL: uqrshl
; CHECK-NOT: uqrshl  r[[REG:[0-9]+]], r[[REG]]
  %1 = tail call i32 @llvm.arm.mve.uqrshl(i32 1, i32 1)
  ret i32 %1
}
