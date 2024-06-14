; RUN: llc -mcpu=corei7 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK: foo
; Make sure we are not trying to use scalar xor on the high bits of the vector.
; CHECK-NOT: xorq
; CHECK: xorl
; CHECK-NEXT: ret

define i32 @foo() {
bb:
  %tmp44.i = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, float 0.000000e+00>
  %0 = bitcast <4 x float> %tmp44.i to i128
  %1 = zext i128 %0 to i512
  %2 = shl nuw nsw i512 %1, 256
  %ins = or i512 %2, 3325764857622480139933400731976840738652108318779753826115024029985671937147149347761402413803120180680770390816681124225944317364750115981129923635970048
  store i512 %ins, ptr undef, align 64
  ret i32 0
}
