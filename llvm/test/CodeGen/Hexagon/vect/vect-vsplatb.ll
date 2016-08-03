; RUN: llc -march=hexagon -disable-hcp < %s | FileCheck %s
; Make sure we build the constant vector <7, 7, 7, 7> with a vsplatb.
; CHECK: vsplatb
@B = common global [400 x i8] zeroinitializer, align 8
@A = common global [400 x i8] zeroinitializer, align 8
@C = common global [400 x i8] zeroinitializer, align 8

define void @run() nounwind {
entry:
  br label %polly.loop_body

polly.loop_after:                                 ; preds = %polly.loop_body
  ret void

polly.loop_body:                                  ; preds = %entry, %polly.loop_body
  %polly.loopiv25 = phi i32 [ 0, %entry ], [ %polly.next_loopiv, %polly.loop_body ]
  %polly.next_loopiv = add i32 %polly.loopiv25, 4
  %p_arrayidx1 = getelementptr [400 x i8], [400 x i8]* @A, i32 0, i32 %polly.loopiv25
  %p_arrayidx = getelementptr [400 x i8], [400 x i8]* @B, i32 0, i32 %polly.loopiv25
  %vector_ptr = bitcast i8* %p_arrayidx to <4 x i8>*
  %_p_vec_full = load <4 x i8>, <4 x i8>* %vector_ptr, align 8
  %mulp_vec = mul <4 x i8> %_p_vec_full, <i8 7, i8 7, i8 7, i8 7>
  %vector_ptr14 = bitcast i8* %p_arrayidx1 to <4 x i8>*
  %_p_vec_full15 = load <4 x i8>, <4 x i8>* %vector_ptr14, align 8
  %addp_vec = add <4 x i8> %_p_vec_full15, %mulp_vec
  store <4 x i8> %addp_vec, <4 x i8>* %vector_ptr14, align 8
  %0 = icmp slt i32 %polly.next_loopiv, 400
  br i1 %0, label %polly.loop_body, label %polly.loop_after
}
