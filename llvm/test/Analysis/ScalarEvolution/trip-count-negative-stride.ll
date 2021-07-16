; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution -scalar-evolution-classify-expressions=0 | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>"  -scalar-evolution-classify-expressions=0 2>&1 | FileCheck %s

; A collection of tests which exercise SCEV's ability to compute trip counts
; for negative steps.

; Case where we wrap the induction variable (without generating poison), and
; thus can't currently compute a trip count.
; CHECK: Determining loop execution counts for: @ult_wrap
; CHECK: Loop %for.body: Unpredictable backedge-taken count
; CHECK: Loop %for.body: Unpredictable max backedge-taken count
define void @ult_wrap() {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %add = add i8 %i.05, 254
  %cmp = icmp ult i8 %add, 255
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; This IV cycles between 0, and 128, never causing the loop to exit
; (This is well defined.)
; CHECK: Determining loop execution counts for: @ult_infinite
; CHECK: Loop %for.body: Unpredictable backedge-taken count
; CHECK: Loop %for.body: Unpredictable max backedge-taken count
define void @ult_infinite() {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %add = add i8 %i.05, 128
  %cmp = icmp ult i8 %add, 255
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Same as ult_infinite, except that the loop is ill defined due to the
; must progress attribute
; CHECK: Determining loop execution counts for: @ult_infinite_ub
; CHECK: Loop %for.body: Unpredictable backedge-taken count
; CHECK: Loop %for.body: Unpredictable max backedge-taken count
define void @ult_infinite_ub() mustprogress {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %add = add i8 %i.05, 128
  %cmp = icmp ult i8 %add, 255
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


; Backedge is not taken
; CHECK: Determining loop execution counts for: @ult_129_not_taken
; CHECK: Loop %for.body: backedge-taken count is 0
; CHECK: Loop %for.body: max backedge-taken count is 0

define void @ult_129_not_taken() {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %add = add i8 %i.05, 129
  %cmp = icmp ult i8 %add, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; A case with a non-constant stride where the backedge is not taken
; CHECK: Determining loop execution counts for: @ult_not_taken
; CHECK: Loop %for.body: Unpredictable backedge-taken count
; CHECK: Loop %for.body: Unpredictable max backedge-taken count

define void @ult_not_taken(i8 %step) {
entry:
  %assume = icmp ult i8 128, %step
  call void @llvm.assume(i1 %assume)
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %add = add i8 %i.05, %step
  %cmp = icmp ult i8 %add, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}



; IV does wrap, and thus causes us to branch on poison.  This loop is
; ill defined.
; CHECK: Determining loop execution counts for: @ult_ub1
; CHECK: Loop %for.body: backedge-taken count is 2
; CHECK: Loop %for.body: max backedge-taken count is 2

define void @ult_ub1() {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 2, %entry ]
  %add = add nuw i8 %i.05, 255
  %cmp = icmp ult i8 %add, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; This loop is ill defined because we violate the nsw flag on the first
; iteration.
; CHECK: Determining loop execution counts for: @ult_ub2
; CHECK: Loop %for.body: backedge-taken count is 0
; CHECK: Loop %for.body: max backedge-taken count is 0

define void @ult_ub2() {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %add = add nsw nuw i8 %i.05, 129
  %cmp = icmp ult i8 %add, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Large stride, poison produced for %add on second iteration, but not
; branched on.
; CHECK: Determining loop execution counts for: @ult_129_preinc
; CHECK: Loop %for.body: backedge-taken count is 1
; CHECK: Loop %for.body: max backedge-taken count is 1

define void @ult_129_preinc() {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %add = add nuw i8 %i.05, 129
  %cmp = icmp ult i8 %i.05, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK: Determining loop execution counts for: @ult_preinc
; CHECK: Loop %for.body: backedge-taken count is 1
; CHECK: Loop %for.body: max backedge-taken count is 1

define void @ult_preinc(i8 %step) {
entry:
  %assume = icmp ult i8 128, %step
  call void @llvm.assume(i1 %assume)
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i8 [ %add, %for.body ], [ 0, %entry ]
  %add = add nuw i8 %i.05, 129
  %cmp = icmp ult i8 %i.05, 128
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare void @llvm.assume(i1)

!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}

