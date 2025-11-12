; REQUIRES: asserts
; RUN: not --crash opt -p loop-vectorize -mcpu=skylake -S %s

target triple = "x86_64-unknown-linux-gnu"

@p = external global [3952 x i8], align 8
@q = external global [3952 x i8], align 8

define void @narrow_store_user_mask_operand(i32 %x) {
entry:
  br label %loop.ph

loop.ph:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.tail ]
  %x.pos = icmp sgt i32 %x, 0
  br i1 %x.pos, label %loop.body, label %loop.tail

loop.body:
  %ld.p = load double, ptr @p
  %gep.q.iv = getelementptr double, ptr @q, i64 %iv
  %gep.q.iv.8 = getelementptr i8, ptr %gep.q.iv, i64 -8
  store double %ld.p, ptr %gep.q.iv.8
  br label %loop.tail

loop.tail:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv, 1
  br i1 %ec, label %exit, label %loop.ph

exit:
  ret void
}
