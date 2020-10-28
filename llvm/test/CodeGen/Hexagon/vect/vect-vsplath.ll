; RUN: llc -march=hexagon < %s | FileCheck %s

; Actually, don't use vsplath.

; CHECK: r[[R0:[0-9]+]] = ##458759
; CHECK: vmpyh(r{{[0-9]+}},r[[R0]])
@B = common global [400 x i16] zeroinitializer, align 8
@A = common global [400 x i16] zeroinitializer, align 8
@C = common global [400 x i16] zeroinitializer, align 8

define void @run() nounwind {
entry:
  br label %polly.loop_body

polly.loop_after:                                 ; preds = %polly.loop_body
  ret void

polly.loop_body:                                  ; preds = %entry, %polly.loop_body
  %polly.loopiv26 = phi i32 [ 0, %entry ], [ %polly.next_loopiv, %polly.loop_body ]
  %polly.next_loopiv = add nsw i32 %polly.loopiv26, 4
  %p_arrayidx1 = getelementptr [400 x i16], [400 x i16]* @A, i32 0, i32 %polly.loopiv26
  %p_arrayidx = getelementptr [400 x i16], [400 x i16]* @B, i32 0, i32 %polly.loopiv26
  %vector_ptr = bitcast i16* %p_arrayidx to <4 x i16>*
  %_p_vec_full = load <4 x i16>, <4 x i16>* %vector_ptr, align 8
  %mulp_vec = mul <4 x i16> %_p_vec_full, <i16 7, i16 7, i16 7, i16 7>
  %vector_ptr15 = bitcast i16* %p_arrayidx1 to <4 x i16>*
  %_p_vec_full16 = load <4 x i16>, <4 x i16>* %vector_ptr15, align 8
  %addp_vec = add <4 x i16> %_p_vec_full16, %mulp_vec
  store <4 x i16> %addp_vec, <4 x i16>* %vector_ptr15, align 8
  %0 = icmp slt i32 %polly.next_loopiv, 400
  br i1 %0, label %polly.loop_body, label %polly.loop_after
}
