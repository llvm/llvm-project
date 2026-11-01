; REQUIRES: x86-registered-target, arm-registered-target
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-runtime -pass-remarks=loop-unroll-and-jam -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VECTOR
; RUN: opt -mtriple=thumbv8m.main-arm-none-eabi -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-runtime -pass-remarks=loop-unroll-and-jam -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NOVECTOR

; Each loop shares the load of b[j] between the copies, so it is profitable to
; jam. But the chained reductions can't be vectorized anymore, which is only
; worth it if they wouldn't have been vectorized anyway.

; Only @strict_fp is jammed if the target can vectorize the other two.
; VECTOR:     remark: {{.*}} unroll and jammed
; VECTOR-NOT: remark:

; Without vector registers nothing is lost.
; NOVECTOR-COUNT-4: remark: {{.*}} unroll and jammed

define void @int_reduction(i64 %n, i64 %m, ptr noalias %c, ptr noalias %b) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %sum = phi i32 [ 0, %outer ], [ %sum.next, %inner ]
  %p = getelementptr inbounds i32, ptr %b, i64 %j
  %v = load i32, ptr %p
  %sum.next = xor i32 %sum, %v
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %sum.lcssa = phi i32 [ %sum.next, %inner ]
  %q = getelementptr inbounds i32, ptr %c, i64 %i
  store i32 %sum.lcssa, ptr %q
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

define void @fast_fp(i64 %n, i64 %m, ptr noalias %c, ptr noalias %b) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %sum = phi fast double [ 0.0, %outer ], [ %sum.next, %inner ]
  %p = getelementptr inbounds double, ptr %b, i64 %j
  %v = load double, ptr %p
  %sum.next = fadd fast double %sum, %v
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %sum.lcssa = phi double [ %sum.next, %inner ]
  %q = getelementptr inbounds double, ptr %c, i64 %i
  store double %sum.lcssa, ptr %q
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

; Without fast-math this has to stay in order, so it can't be vectorized.
define void @strict_fp(i64 %n, i64 %m, ptr noalias %c, ptr noalias %b) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %sum = phi double [ 0.0, %outer ], [ %sum.next, %inner ]
  %p = getelementptr inbounds double, ptr %b, i64 %j
  %v = load double, ptr %p
  %sum.next = fadd double %sum, %v
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %sum.lcssa = phi double [ %sum.next, %inner ]
  %q = getelementptr inbounds double, ptr %c, i64 %i
  store double %sum.lcssa, ptr %q
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}

define void @fixed_order_recurrence(i64 %n, i64 %m, ptr noalias %c, ptr noalias %b) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %prev = phi i32 [ 0, %outer ], [ %v, %inner ]
  %p = getelementptr inbounds i32, ptr %b, i64 %j
  %v = load i32, ptr %p
  %add = add i32 %v, %prev
  %q = getelementptr inbounds [1000 x i32], ptr %c, i64 %i, i64 %j
  store i32 %add, ptr %q
  %j.next = add nuw nsw i64 %j, 1
  %inner.cond = icmp eq i64 %j.next, %m
  br i1 %inner.cond, label %outer.latch, label %inner

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %outer.cond = icmp eq i64 %i.next, %n
  br i1 %outer.cond, label %exit, label %outer

exit:
  ret void
}
