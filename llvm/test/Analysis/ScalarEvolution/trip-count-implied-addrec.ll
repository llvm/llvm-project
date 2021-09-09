; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>"  -scalar-evolution-classify-expressions=0 2>&1 | FileCheck %s

; A collection of tests that show we can use facts about an exit test to
; infer tighter bounds on an IV, and thus refine an IV into an addrec. The
; basic tactic being used is proving NW from exit structure and then
; implying NUW/NSW.  Once NSW/NUW is inferred, we can derive addrecs from
; the zext/sext cases that we couldn't at initial SCEV construction.

@G = external global i8

; CHECK-LABEL: Determining loop execution counts for: @nw_implies_nuw
; CHECK: Loop %for.body: backedge-taken count is %n
; CHECK: Loop %for.body: max backedge-taken count is -1
define void @nw_implies_nuw(i16 %n) mustprogress {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i8 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i8 %iv, 1
  %zext = zext i8 %iv to i16
  %cmp = icmp ult i16 %zext, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Determining loop execution counts for: @neg_nw_nuw
; CHECK: Loop %for.body: Unpredictable backedge-taken count
; CHECK: Loop %for.body: Unpredictable max backedge-taken count
define void @neg_nw_nuw(i16 %n) mustprogress {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i8 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i8 %iv, -1
  %zext = zext i8 %iv to i16
  %cmp = icmp ult i16 %zext, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Determining loop execution counts for: @nw_implies_nsw
; CHECK: Loop %for.body: Unpredictable backedge-taken count
; CHECK: Loop %for.body: Unpredictable max backedge-taken count
define void @nw_implies_nsw(i16 %n) mustprogress {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i8 [ %iv.next, %for.body ], [ -128, %entry ]
  %iv.next = add i8 %iv, 1
  %zext = sext i8 %iv to i16
  %cmp = icmp slt i16 %zext, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Determining loop execution counts for: @neg_nw_nsw
; CHECK: Loop %for.body: Unpredictable backedge-taken count
; CHECK: Loop %for.body: Unpredictable max backedge-taken count
define void @neg_nw_nsw(i16 %n) mustprogress {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i8 [ %iv.next, %for.body ], [ -128, %entry ]
  %iv.next = add i8 %iv, -1
  %zext = sext i8 %iv to i16
  %cmp = icmp slt i16 %zext, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


; CHECK-LABEL: Determining loop execution counts for: @actually_infinite
; CHECK: Loop %for.body: Unpredictable backedge-taken count
; CHECK: Loop %for.body: Unpredictable max backedge-taken count
define void @actually_infinite() {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i8 [ %iv.next, %for.body ], [ 0, %entry ]
  store volatile i8 %iv, i8* @G
  %iv.next = add i8 %iv, 1
  %zext = zext i8 %iv to i16
  %cmp = icmp ult i16 %zext, 257
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare void @llvm.assume(i1)

