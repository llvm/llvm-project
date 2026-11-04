; RUN: opt -passes=loop-vectorize -disable-output %s

; Make sure we do not crash when the loop contains a conditional induction
; whose use accesses memory through a scalable vector type. Scalable types
; have no fixed allocation size, so the compressed-pointer optimization is
; not applicable and legality should bail out gracefully.

define void @f(ptr %dst, ptr %c, i64 %n) {
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %p = phi ptr [ %dst, %entry ], [ %p.1, %latch ]
  %gep.c = getelementptr i32, ptr %c, i64 %iv
  %cv = load i32, ptr %gep.c
  %cmp = icmp ne i32 %cv, 0
  br i1 %cmp, label %then, label %latch
then:
  store <vscale x 4 x i32> zeroinitializer, ptr %p
  %p.next = getelementptr i8, ptr %p, i64 16
  br label %latch
latch:
  %p.1 = phi ptr [ %p.next, %then ], [ %p, %loop ]
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop
exit:
  ret void
}
