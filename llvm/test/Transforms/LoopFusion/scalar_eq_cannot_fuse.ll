; RUN: opt -passes=loop-simplify,loop-fusion -stats -disable-output < %s 2>&1 | FileCheck %s --check-prefix=STAT
; RUN: opt -passes=loop-simplify,loop-fusion -pass-remarks-missed=loop-fusion -disable-output < %s 2>&1 | FileCheck %s
; STAT-NOT: loop-fusion{{.*}} - Loops fused
; STAT: 2 loop-fusion{{.*}} - Dependencies prevent fusion

; Negative tests: the scalar EQ dependence change must not fuse these patterns.

; Loop 1 stores A[i]; loop 2 loads A[i + 1] — forward loop-carried dependence.
; CHECK: remark: {{.*}}store_then_load_next{{.*}}: Dependencies prevent fusion

define void @store_then_load_next(ptr noalias nocapture %A, i64 %n, float %w0) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %gep1 = getelementptr inbounds float, ptr %A, i64 %i1
  store float %w0, ptr %gep1, align 4
  br label %loop1.latch

loop1.latch:
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2.latch ]
  %next = add nuw nsw i64 %i2, 1
  %gep2 = getelementptr inbounds float, ptr %A, i64 %next
  %v = load float, ptr %gep2, align 4
  %gep3 = getelementptr inbounds float, ptr %A, i64 %i2
  store float %v, ptr %gep3, align 4
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

; Cross-pointer store-then-accumulate without noalias on the output pointer.
; CHECK: remark: {{.*}}cross_pointer_may_alias{{.*}}: Dependencies prevent fusion

define void @cross_pointer_may_alias(ptr nocapture %hist, ptr nocapture readonly %input, i64 %n, float %w0, float %w1) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %a = getelementptr inbounds float, ptr %input, i64 %i1
  %v0 = load float, ptr %a, align 4
  %p = fmul float %w0, %v0
  %h1 = getelementptr inbounds float, ptr %hist, i64 %i1
  store float %p, ptr %h1, align 4
  br label %loop1.latch

loop1.latch:
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2.latch ]
  %b = getelementptr inbounds float, ptr %input, i64 %i2
  %v1 = load float, ptr %b, align 4
  %h2 = getelementptr inbounds float, ptr %hist, i64 %i2
  %old = load float, ptr %h2, align 4
  %p1 = fmul float %w1, %v1
  %sum = fadd float %old, %p1
  store float %sum, ptr %h2, align 4
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}
