; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that we generate a hardware loop instruction.
; CHECK: endloop0

@A = common global [400 x i8] zeroinitializer, align 8
@B = common global [400 x i8] zeroinitializer, align 8
@C = common global [400 x i8] zeroinitializer, align 8

define void @run() nounwind {
entry:
  br label %polly.loop_body

polly.loop_after:                                 ; preds = %polly.loop_body
  ret void

polly.loop_body:                                  ; preds = %entry, %polly.loop_body
  %polly.loopiv16 = phi i32 [ 0, %entry ], [ %polly.next_loopiv, %polly.loop_body ]
  %polly.next_loopiv = add i32 %polly.loopiv16, 4
  %p_vector_iv14 = or i32 %polly.loopiv16, 1
  %p_vector_iv3 = add i32 %p_vector_iv14, 1
  %p_vector_iv415 = or i32 %polly.loopiv16, 3
  %p_arrayidx = getelementptr [400 x i8], ptr @A, i32 0, i32 %polly.loopiv16
  %p_arrayidx5 = getelementptr [400 x i8], ptr @A, i32 0, i32 %p_vector_iv14
  %p_arrayidx6 = getelementptr [400 x i8], ptr @A, i32 0, i32 %p_vector_iv3
  %p_arrayidx7 = getelementptr [400 x i8], ptr @A, i32 0, i32 %p_vector_iv415
  store i8 123, ptr %p_arrayidx, align 1
  store i8 123, ptr %p_arrayidx5, align 1
  store i8 123, ptr %p_arrayidx6, align 1
  store i8 123, ptr %p_arrayidx7, align 1
  %0 = icmp slt i32 %polly.next_loopiv, 400
  br i1 %0, label %polly.loop_body, label %polly.loop_after
}
