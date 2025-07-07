; RUN: llc -mtriple=hexagon < %s
;
; Used to fail with "Cannot select: v2i32,ch = load 0x16c5890, 0x16f76e0, 0x16f76e0<LD2[undef](align=8), sext from v2i8>"

target triple = "hexagon-unknown-linux-gnu"

define void @foo(ptr %p) nounwind {
entry:
  br label %polly.loop_header

polly.loop_after:                                 ; preds = %polly.loop_header
  unreachable

polly.loop_header:                                ; preds = %polly.loop_body, %entry
  %0 = icmp sle i32 undef, 63
  br i1 %0, label %polly.loop_body, label %polly.loop_after

polly.loop_body:                                  ; preds = %polly.loop_header
  %_p_vec_full = load <2 x i8>, ptr %p, align 8
  %1 = sext <2 x i8> %_p_vec_full to <2 x i32>
  %p_vec = mul <2 x i32> %1, <i32 3, i32 3>
  %mulp_vec = add <2 x i32> %p_vec, <i32 21, i32 21>
  store <2 x i32> %mulp_vec, ptr undef, align 8
  br label %polly.loop_header
}
