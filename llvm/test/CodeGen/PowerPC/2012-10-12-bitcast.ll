; RUN: llc -verify-machineinstrs -mattr=-vsx -mattr=+altivec -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mattr=+vsx -mattr=+altivec -mcpu=pwr7 < %s | FileCheck -check-prefix=CHECK-VSX %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @test(<16 x i8> %v) nounwind {
entry:
  %0 = bitcast <16 x i8> %v to i128
  %1 = lshr i128 %0, 96
  %2 = trunc i128 %1 to i32
  ret i32 %2
}

; Verify that bitcast handles big-endian platforms correctly
; by checking we load the result from the correct offset

; CHECK: addi [[REGISTER:[0-9]+]], 1, -16
; CHECK: stvx 2, 0, [[REGISTER]]
; CHECK: lwz 3, -16(1)
; CHECK: blr

; CHECK-VSX: addi [[REGISTER:[0-9]+]], 1, -16
; CHECK-VSX: stxvd2x 34, 0, [[REGISTER]]
; CHECK-VSX: lwz 3, -16(1)
; CHECK-VSX: blr
