; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; PR13228
define void @f(i1 %arg) nounwind uwtable readnone {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %c.0 = phi i8 [ 1, %entry ], [ 0, %for.cond ]
  %i.0 = phi i8 [ 0, %entry ], [ %inc, %for.cond ]
  %lnot = icmp eq i8 %i.0, 0
  %inc = add i8 %i.0, 1
  br i1 %lnot, label %for.cond, label %while.cond

while.cond:                                       ; preds = %while.body, %for.cond
  %b.2 = phi i8 [ %add, %while.body ], [ 0, %for.cond ]
  br i1 %arg, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %add = add i8 %b.2, %c.0
  %tobool7 = icmp eq i8 %add, 0
  br i1 %tobool7, label %while.end, label %while.cond

while.end:                                        ; preds = %while.body, %while.cond
  ret void
}
;CHECK: Loop %while.cond: <multiple exits> Unpredictable backedge-taken count.
