; REQUIRES: asserts

; RUN: opt -S -passes=loop-fusion -debug-only=loop-fusion -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-DA

define void @loop_invariant(i32 %N) {
; CHECK-DA: Performing Loop Fusion on function loop_invariant
; CHECK-DA: Safe to fuse due to a loop-invariant output dependency
;
pre1:
  %ptr = alloca i32, align 4
  br label %body1

body1:  ; preds = %pre1, %body1
  %i = phi i32 [%i_next, %body1], [0, %pre1]
  %i_next = add i32 1, %i
  %cond = icmp ne i32 %i, %N
  store i32 3, ptr %ptr
  br i1 %cond, label %body1, label %pre2

pre2:
  br label %body2

body2:  ; preds = %pre2, %body2
  %i2 = phi i32 [%i_next2, %body2], [0, %pre2]
  %i_next2 = add i32 1, %i2
  %cond2 = icmp ne i32 %i2, %N
  store i32 3, ptr %ptr
  br i1 %cond2, label %body2, label %exit

exit:
  ret void
}

define void @anti_loop_invariant(i32 %N) {
; CHECK-DA: Performing Loop Fusion on function anti_loop_invariant
; CHECK-DA: Memory dependencies do not allow fusion!
;
pre1:
  %ptr = alloca i32, align 4
  store i32 1, ptr %ptr
  br label %body1

body1:  ; preds = %pre1, %body1
  %i = phi i32 [%i_next, %body1], [0, %pre1]
  %i_next = add i32 1, %i
  %cond = icmp ne i32 %i, %N
  %v = load i32, ptr %ptr
  br i1 %cond, label %body1, label %pre2

pre2:
  br label %body2

body2:  ; preds = %pre2, %body2
  %i2 = phi i32 [%i_next2, %body2], [0, %pre2]
  %i_next2 = add i32 1, %i2
  %cond2 = icmp ne i32 %i2, %N
  store i32 3, ptr %ptr
  br i1 %cond2, label %body2, label %exit

exit:
  ret void
}

; Test idempotent store-store pairs: both loops write the same value to
; provably different underlying objects. Safe to fuse despite MayAlias.

define void @idempotent_same_arg(ptr %a, ptr %b, i32 %n, i32 %val) {
; CHECK-DA: Performing Loop Fusion on function idempotent_same_arg
; CHECK-DA: Fusion is performed
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body.preheader, label %for.cond2.preheader

for.body.preheader:
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %for.body.preheader ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %val, ptr %gep.a, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exit = icmp eq i64 %iv.next, %wide.trip.count
  br i1 %exit, label %for.cond2.preheader, label %for.body

for.cond2.preheader:
  %cmp2 = icmp sgt i32 %n, 0
  br i1 %cmp2, label %for.body5.preheader, label %for.cond.cleanup4

for.body5.preheader:
  %wide.trip.count2 = zext nneg i32 %n to i64
  br label %for.body5

for.body5:
  %iv2 = phi i64 [ 0, %for.body5.preheader ], [ %iv2.next, %for.body5 ]
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv2
  store i32 %val, ptr %gep.b, align 4
  %iv2.next = add nuw nsw i64 %iv2, 1
  %exit2 = icmp eq i64 %iv2.next, %wide.trip.count2
  br i1 %exit2, label %for.cond.cleanup4, label %for.body5

for.cond.cleanup4:
  ret void
}

define void @idempotent_same_expr(ptr %a, ptr %b, i32 %n, i32 %val, i32 %unknown) {
; CHECK-DA: Performing Loop Fusion on function idempotent_same_expr
; CHECK-DA: Fusion is performed
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body.lr.ph, label %for.cond2.preheader

for.body.lr.ph:
  %add = add nsw i32 %unknown, %val
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %for.body.lr.ph ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %gep.a, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exit = icmp eq i64 %iv.next, %wide.trip.count
  br i1 %exit, label %for.cond2.preheader, label %for.body

for.cond2.preheader:
  %cmp2 = icmp sgt i32 %n, 0
  br i1 %cmp2, label %for.body5.lr.ph, label %for.cond.cleanup4

for.body5.lr.ph:
  %add6 = add nsw i32 %unknown, %val
  %wide.trip.count2 = zext nneg i32 %n to i64
  br label %for.body5

for.body5:
  %iv2 = phi i64 [ 0, %for.body5.lr.ph ], [ %iv2.next, %for.body5 ]
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv2
  store i32 %add6, ptr %gep.b, align 4
  %iv2.next = add nuw nsw i64 %iv2, 1
  %exit2 = icmp eq i64 %iv2.next, %wide.trip.count2
  br i1 %exit2, label %for.cond.cleanup4, label %for.body5

for.cond.cleanup4:
  ret void
}

define void @idempotent_different_values(ptr %a, ptr %b, i32 %n, i32 %t) {
; CHECK-DA: Performing Loop Fusion on function idempotent_different_values
; CHECK-DA: Memory dependencies do not allow fusion!
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body.preheader, label %for.cond2.preheader

for.body.preheader:
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %for.body.preheader ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %t, ptr %gep.a, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exit = icmp eq i64 %iv.next, %wide.trip.count
  br i1 %exit, label %for.cond2.preheader, label %for.body

for.cond2.preheader:
  %cmp2 = icmp sgt i32 %n, 0
  br i1 %cmp2, label %for.body5.preheader, label %for.cond.cleanup4

for.body5.preheader:
  %wide.trip.count2 = zext nneg i32 %n to i64
  br label %for.body5

for.body5:
  %iv2 = phi i64 [ 0, %for.body5.preheader ], [ %iv2.next, %for.body5 ]
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv2
  store i32 42, ptr %gep.b, align 4
  %iv2.next = add nuw nsw i64 %iv2, 1
  %exit2 = icmp eq i64 %iv2.next, %wide.trip.count2
  br i1 %exit2, label %for.cond.cleanup4, label %for.body5

for.cond.cleanup4:
  ret void
}
