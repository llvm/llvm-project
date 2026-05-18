; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: vaslh

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon-unknown-linux-gnu"

define void @foo(ptr nocapture %v) nounwind {
entry:
  %p_arrayidx = getelementptr i16, ptr %v, i32 4
  %_p_vec_full = load <4 x i16>, ptr %p_arrayidx, align 2
  %_high_half = shufflevector <4 x i16> %_p_vec_full, <4 x i16> undef, <2 x i32> <i32 2, i32 3>
  %_low_half = shufflevector <4 x i16> %_p_vec_full, <4 x i16> undef, <2 x i32> <i32 0, i32 1>
  %0 = sext <2 x i16> %_low_half to <2 x i32>
  %1 = sext <2 x i16> %_high_half to <2 x i32>
  %shr6p_vec = shl <2 x i32> %0, <i32 2, i32 2>
  %shr6p_vec19 = shl <2 x i32> %1, <i32 2, i32 2>
  %addp_vec = add <2 x i32> %shr6p_vec, <i32 34, i32 34>
  %addp_vec20 = add <2 x i32> %shr6p_vec19, <i32 34, i32 34>
  %_p_vec_full22 = load <4 x i16>, ptr %v, align 2
  %_high_half23 = shufflevector <4 x i16> %_p_vec_full22, <4 x i16> undef, <2 x i32> <i32 2, i32 3>
  %_low_half24 = shufflevector <4 x i16> %_p_vec_full22, <4 x i16> undef, <2 x i32> <i32 0, i32 1>
  %2 = zext <2 x i16> %_low_half24 to <2 x i32>
  %3 = zext <2 x i16> %_high_half23 to <2 x i32>
  %add3p_vec = add <2 x i32> %addp_vec, %2
  %add3p_vec25 = add <2 x i32> %addp_vec20, %3
  %4 = trunc <2 x i32> %add3p_vec to <2 x i16>
  %5 = trunc <2 x i32> %add3p_vec25 to <2 x i16>
  %_combined_vec = shufflevector <2 x i16> %4, <2 x i16> %5, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i16> %_combined_vec, ptr %v, align 2
  ret void
}
