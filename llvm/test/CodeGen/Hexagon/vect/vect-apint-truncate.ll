; RUN: llc -mtriple=hexagon < %s
; Used to fail with "Invalid APInt Truncate request".
; Used to fail with "Cannot select: 0x596010: v2i32 = sign_extend_inreg".

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

define void @foo() nounwind {
entry:
  br label %polly.loop_header

polly.loop_after:                                 ; preds = %polly.loop_header
  unreachable

polly.loop_header:                                ; preds = %polly.loop_body, %entry
  %0 = icmp sle i32 undef, 63
  br i1 %0, label %polly.loop_body, label %polly.loop_after

polly.loop_body:                                  ; preds = %polly.loop_header
  %_p_vec_full = load <4 x i8>, ptr undef, align 8
  %1 = sext <4 x i8> %_p_vec_full to <4 x i32>
  %p_vec = mul <4 x i32> %1, <i32 3, i32 3, i32 3, i32 3>
  %mulp_vec = add <4 x i32> %p_vec, <i32 21, i32 21, i32 21, i32 21>
  store <4 x i32> %mulp_vec, ptr undef, align 8
  br label %polly.loop_header
}
