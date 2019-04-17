; RUN: opt < %s -loop-vectorize -S | FileCheck %s

; Out of the LCSSA form we could have 'phi i32 [ loop-invariant, %for.inc.2.i ]'
; but the IR Verifier requires for PHI one entry for each predecessor of
; it's parent basic block. The original PR14725 solution for the issue just
; added 'undef' for an predecessor BB and which is not correct. We copy the real
; value for another predecessor instead of bringing 'undef'.

; CHECK-LABEL: for.cond.preheader:
; CHECK: %e.0.ph = phi i32 [ 0, %if.end.2.i ], [ 0, %middle.block ]

; Function Attrs: nounwind uwtable
define void @main() #0 {
entry:
  br label %for.cond1.preheader.i

for.cond1.preheader.i:                            ; preds = %if.end.2.i, %entry
  %c.06.i = phi i32 [ 0, %entry ], [ %inc5.i, %if.end.2.i ]
  %tobool.i = icmp ne i32 undef, 0
  br label %if.end.2.i

if.end.2.i:                                       ; preds = %for.cond1.preheader.i
  %inc5.i = add nsw i32 %c.06.i, 1
  %cmp.i = icmp slt i32 %inc5.i, 16
  br i1 %cmp.i, label %for.cond1.preheader.i, label %for.cond.preheader

for.cond.preheader:                               ; preds = %if.end.2.i
  %e.0.ph = phi i32 [ 0, %if.end.2.i ]
  unreachable
}
