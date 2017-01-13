; RUN: opt < %s -S -debug-only=loop-unroll -loop-unroll -unroll-allow-peeling 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure we use the profile information correctly to peel-off 3 iterations
; from the loop, and update the branch weights for the peeled loop properly.

; CHECK: Loop Unroll: F[basic]
; CHECK: PEELING loop %for.body with iteration count 3!
; CHECK: Loop Unroll: F[optsize]
; CHECK-NOT: PEELING

; CHECK-LABEL: @basic
; CHECK: br i1 %{{.*}}, label %[[NEXT0:.*]], label %for.cond.for.end_crit_edge, !prof !1
; CHECK: [[NEXT0]]:
; CHECK: br i1 %{{.*}}, label %[[NEXT1:.*]], label %for.cond.for.end_crit_edge, !prof !2
; CHECK: [[NEXT1]]:
; CHECK: br i1 %{{.*}}, label %[[NEXT2:.*]], label %for.cond.for.end_crit_edge, !prof !3
; CHECK: [[NEXT2]]:
; CHECK: br i1 %{{.*}}, label %for.body, label %{{.*}}, !prof !4

define void @basic(i32* %p, i32 %k) #0 !prof !0 {
entry:
  %cmp3 = icmp slt i32 0, %k
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %p.addr.04 = phi i32* [ %p, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %p.addr.04, i32 1
  store i32 %i.05, i32* %p.addr.04, align 4
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %inc, %k
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !prof !1

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

; We don't want to peel loops when optimizing for size.
; CHECK-LABEL: @optsize
; CHECK: for.body.lr.ph:
; CHECK-NEXT: br label %for.body
; CHECK: for.body:
; CHECK-NOT: br
; CHECK: br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge
define void @optsize(i32* %p, i32 %k) #1 !prof !0 {
entry:
  %cmp3 = icmp slt i32 0, %k
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %p.addr.04 = phi i32* [ %p, %for.body.lr.ph ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %p.addr.04, i32 1
  store i32 %i.05, i32* %p.addr.04, align 4
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %inc, %k
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !prof !1

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind optsize }

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 3001, i32 1001}

;CHECK: !1 = !{!"branch_weights", i32 900, i32 101}
;CHECK: !2 = !{!"branch_weights", i32 540, i32 360}
;CHECK: !3 = !{!"branch_weights", i32 162, i32 378}
;CHECK: !4 = !{!"branch_weights", i32 1399, i32 162}

