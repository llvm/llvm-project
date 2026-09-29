; RUN: llc -mcpu=cortex-a8 < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "armv7-apple-darwin10"

%struct.int16x8_t = type { <8 x i16> }
%struct.int16x8x2_t = type { [2 x %struct.int16x8_t] }

define void @t(ptr noalias nocapture sret(%struct.int16x8x2_t) %agg.result, <8 x i16> %tmp.0, ptr nocapture %dst) nounwind {
entry:
;CHECK: vtrn.16
  %0 = shufflevector <8 x i16> %tmp.0, <8 x i16> undef, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  %1 = shufflevector <8 x i16> %tmp.0, <8 x i16> undef, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  %agg.result1218.0 = getelementptr %struct.int16x8x2_t, ptr %agg.result, i32 0, i32 0, i32 0, i32 0 ; <ptr>
  store <8 x i16> %0, ptr %agg.result1218.0, align 16
  %agg.result12.1.0 = getelementptr %struct.int16x8x2_t, ptr %agg.result, i32 0, i32 0, i32 1, i32 0 ; <ptr>
  store <8 x i16> %1, ptr %agg.result12.1.0, align 16
  ret void
}

; Radar 8290937: Ignore undef shuffle indices.
; CHECK: t2
; CHECK: vtrn.16
define void @t2(ptr nocapture %ptr, <4 x i16> %a.0, <4 x i16> %b.0) nounwind {
entry:
  %0 = shufflevector <4 x i16> %a.0, <4 x i16> undef, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 undef, i32 undef, i32 undef, i32 undef>
  %1 = shufflevector <4 x i16> %a.0, <4 x i16> undef, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  store <8 x i16> %0, ptr %ptr, align 16
  %ptr20.1.0 = getelementptr inbounds %struct.int16x8x2_t, ptr %ptr, i32 0, i32 0, i32 1, i32 0
  store <8 x i16> %1, ptr %ptr20.1.0, align 16
  ret void
}
