; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:   -force-target-supports-gather-scatter-ops -pass-remarks-analysis=loop-vectorize \
; RUN:   -disable-output %s 2>&1 | FileCheck %s --implicit-check-not=remark

; swifterror values may only be used as the pointer operand of loads and stores
; or as swifterror call arguments. Loops using them are not vectorized, as
; vectorization may introduce other uses, e.g. vectors of pointers.

; Would be widened to a scatter with a broadcast swifterror pointer.
; CHECK: remark: <unknown>:0:0: loop not vectorized: swifterror value cannot be vectorized
define void @predicated_store(ptr %c, ptr swifterror %err, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gep = getelementptr i1, ptr %c, i64 %iv
  %l = load i1, ptr %gep
  br i1 %l, label %then, label %latch

then:
  store ptr null, ptr %err
  br label %latch

latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; Would be vectorized correctly, as the store stays scalar, but is rejected to
; not depend on predication and widening decisions.
; CHECK: remark: <unknown>:0:0: loop not vectorized: swifterror value cannot be vectorized
define void @uniform_store(ptr swifterror %err, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  store ptr null, ptr %err
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; The alloca would be replicated per lane and packed into a vector.
; CHECK: remark: <unknown>:0:0: loop not vectorized: swifterror value cannot be vectorized
define void @alloca_in_loop(i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %err = alloca swifterror ptr
  store ptr null, ptr %err
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
