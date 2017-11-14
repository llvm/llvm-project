; RUN: opt -scalar-evolution -loop-deletion -simplifycfg -analyze < %s | FileCheck %s --check-prefix=CHECK-ANALYSIS-1
; RUN: opt -analyze -scalar-evolution < %s | FileCheck %s --check-prefix=CHECK-ANALYSIS-2

define i32 @pr34538() local_unnamed_addr #0 {
; CHECK-ANALYSIS-1: Loop %do.body: backedge-taken count is 10000
; CHECK-ANALYSIS-1: Loop %do.body: max backedge-taken count is 10000
; CHECK-ANALYSIS-1: Loop %do.body: Predicated backedge-taken count is 10000
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %start.0 = phi i32 [ 0, %entry ], [ %inc.start.0, %do.body ]
  %cmp = icmp slt i32 %start.0, 10000
  %inc = zext i1 %cmp to i32
  %inc.start.0 = add nsw i32 %start.0, %inc
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  ret i32 0
}


define i32 @foo() {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %start.0 = phi i32 [ 0, %entry ], [ %inc.start.0, %do.body ]
  %cmp = icmp slt i32 %start.0, 10000
  %select_ext = select i1 %cmp, i32 2 , i32 1
  %inc.start.0 = add nsw i32 %start.0, %select_ext
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  ret i32 0
; CHECK-ANALYSIS-2: Loop %do.body: backedge-taken count is 5000
; CHECK-ANALYSIS-2: Loop %do.body: max backedge-taken count is 5000
; CHECK-ANALYSIS-2: Loop %do.body: Predicated backedge-taken count is 5000
}
