; RUN: opt < %s -loop-vectorize -debug-only=loop-vectorize -S -disable-output 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure LV legal bails out when the exiting block != loop latch.
; CHECK-LABEL: "latch_is_not_exiting"
; CHECK: LV: Not vectorizing: The exiting block is not the loop latch.
define i32 @latch_is_not_exiting() {
entry:
  br label %for.body

for.body:
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ], [%inc, %for.second]
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, 16
  br i1 %cmp, label %for.body, label %for.second

for.second:
  %cmps = icmp sgt i32 %inc, 16
  br i1 %cmps, label %for.body, label %for.end

for.end:
  ret i32 0
}

; Make sure LV legal bails out when there is no exiting block
; CHECK-LABEL: "no_exiting_block"
; CHECK: LV: Not vectorizing: The loop must have an exiting block.
define i32 @no_exiting_block() {
entry:
  br label %for.body

for.body:
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ], [%inc, %for.second]
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, 16
  br i1 %cmp, label %for.body, label %for.second

for.second:
  br label %for.body
}

; Make sure LV legal bails out when there is a non-int, non-ptr phi
; CHECK-LABEL: "invalid_phi_types"
; CHECK: LV: Not vectorizing: Found a non-int non-pointer PHI.
define i32 @invalid_phi_types() {
entry:
  br label %for.body

for.body:
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %vec.sum.02 = phi <2 x i32> [ zeroinitializer, %entry ], [ <i32 8, i32 8>, %for.body ]
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, 16
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret i32 0
}

; D40973
; Make sure LV legal bails out when the loop doesn't have a legal pre-header.
; CHECK-LABEL: "inc"
; CHECK: LV: Not vectorizing: Loop doesn't have a legal pre-header.
define void @inc(i32 %n, i8* %P) {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %BB1, label %BB2

BB1:
  indirectbr i8* %P, [label %.lr.ph]

BB2:
  br label %.lr.ph

.lr.ph:
  %indvars.iv = phi i32 [ %indvars.iv.next, %.lr.ph ], [ 0, %BB1 ], [ 0, %BB2 ]
  %indvars.iv.next = add i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:
  ret void
}
