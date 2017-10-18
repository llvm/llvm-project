; RUN: opt -S -scalar-evolution -loop-deletion -simplifycfg -analyze < %s | FileCheck %s --check-prefix=CHECK-ANALYSIS

define i32 @foo() local_unnamed_addr #0 {
; CHECK-ANALYSIS: Loop %do.body: backedge-taken count is 10000
; CHECK-ANALYSIS: Loop %do.body: max backedge-taken count is 10000
; CHECK-ANALYSIS: Loop %do.body: Predicated backedge-taken count is 10000
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
