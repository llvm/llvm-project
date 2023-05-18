; RUN: llc -O3 %s -o - | FileCheck %s
; ModuleID = 'fo.c'
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:32-n8:16:32-S64"
target triple = "thumbv7-none-linux-gnueabi"

; CHECK: vpush
; CHECK: vpop

define void @foo(ptr nocapture %A) #0 {
  %1= bitcast ptr %A to ptr
  %2 = tail call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4.v4f32.p0(ptr %1, i32 4)
  %3 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 0
  %divp_vec = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %3
  %4 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 1
  %div3p_vec = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %4
  %5 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 2
  %div8p_vec = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %5
  %6 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %2, 3
  %div13p_vec = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %6
  tail call void @llvm.arm.neon.vst4.p0.v4f32(ptr %1, <4 x float> %divp_vec, <4 x float> %div3p_vec, <4 x float> %div8p_vec, <4 x float> %div13p_vec, i32 4)
 ret void
}

; Function Attrs: nounwind
declare i32 @llvm.annotation.i32(i32, ptr, ptr, i32) #1

; Function Attrs: nounwind readonly

; Function Attrs: nounwind
declare void @llvm.arm.neon.vst4.p0.v4f32(ptr, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32) #1
declare { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4.v4f32.p0(ptr, i32) #2

; Function Attrs: nounwind

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }

!llvm.ident = !{!0}

!0 = !{!"Snapdragon LLVM ARM Compiler 3.4"}
!1 = !{!1}
