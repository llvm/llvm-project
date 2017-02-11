; RUN: llc -march=hexagon -mcpu=hexagonv5 -disable-hsdr < %s | FileCheck %s
; This one should generate a combine with two immediates.
; CHECK: combine(#7,#7)
@B = common global [400 x i32] zeroinitializer, align 8
@A = common global [400 x i32] zeroinitializer, align 8
@C = common global [400 x i32] zeroinitializer, align 8

define void @run() nounwind {
entry:
  br label %polly.loop_body

polly.loop_after:                                 ; preds = %polly.loop_body
  ret void

polly.loop_body:                                  ; preds = %entry, %polly.loop_body
  %polly.loopiv23 = phi i32 [ 0, %entry ], [ %polly.next_loopiv, %polly.loop_body ]
  %polly.next_loopiv = add nsw i32 %polly.loopiv23, 4
  %p_arrayidx1 = getelementptr [400 x i32], [400 x i32]* @A, i32 0, i32 %polly.loopiv23
  %p_arrayidx = getelementptr [400 x i32], [400 x i32]* @B, i32 0, i32 %polly.loopiv23
  %vector_ptr = bitcast i32* %p_arrayidx to <4 x i32>*
  %_p_vec_full = load <4 x i32>, <4 x i32>* %vector_ptr, align 8
  %mulp_vec = mul <4 x i32> %_p_vec_full, <i32 7, i32 7, i32 7, i32 7>
  %vector_ptr12 = bitcast i32* %p_arrayidx1 to <4 x i32>*
  %_p_vec_full13 = load <4 x i32>, <4 x i32>* %vector_ptr12, align 8
  %addp_vec = add <4 x i32> %_p_vec_full13, %mulp_vec
  store <4 x i32> %addp_vec, <4 x i32>* %vector_ptr12, align 8
  %0 = icmp slt i32 %polly.next_loopiv, 400
  br i1 %0, label %polly.loop_body, label %polly.loop_after
}
