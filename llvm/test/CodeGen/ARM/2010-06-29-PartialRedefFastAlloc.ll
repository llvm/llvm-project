; RUN: llc < %s -O0 -mcpu=cortex-a8 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This tests the fast register allocator's handling of partial redefines:
;
;      %reg1028:dsub_0<def>, %reg1028:dsub_1<def> = VLD1q64 %reg1025...
;      %reg1030:dsub_1<def> = COPY %reg1028:dsub_0<kill>
;
; %reg1028 gets allocated %Q0, and if %reg1030 is reloaded for the partial
; redef, it cannot also get %Q0.

; CHECK: vld1.64 {d16, d17}, [r{{.}}]
; CHECK-NOT: vld1.64 {d16, d17}
; CHECK: vmov.f64

define i32 @test(i8* %arg) nounwind {
entry:
 %0 = call <2 x i64> @llvm.arm.neon.vld1.v2i64(i8* %arg, i32 1)
 %1 = shufflevector <2 x i64> undef, <2 x i64> %0, <2 x i32> <i32 1, i32 2>
 store <2 x i64> %1, <2 x i64>* undef, align 16
 ret i32 undef
}

declare <2 x i64> @llvm.arm.neon.vld1.v2i64(i8*, i32) nounwind readonly
