; RUN: opt %loadPolly -polly-codegen < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
define void @Reflection_coefficients(ptr %r) nounwind {
bb20:
  %indvar3.lcssa20.reload = load i64, ptr undef
  %tmp = mul i64 %indvar3.lcssa20.reload, -1
  %tmp5 = add i64 %tmp, 8
  br label %bb22

bb21:                                             ; preds = %bb22
  store i16 0, ptr %r, align 2
  %indvar.next = add i64 %indvar, 1
  br label %bb22

bb22:                                             ; preds = %bb21, %bb20
  %indvar = phi i64 [ %indvar.next, %bb21 ], [ 0, %bb20 ]
  %exitcond = icmp ne i64 %indvar, %tmp5
  br i1 %exitcond, label %bb21, label %return

return:                                           ; preds = %bb22
  ret void
}
