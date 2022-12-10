; RUN: opt < %s -passes=reassociate -disable-output
; PR13041

define void @foo() {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %b.0 = phi i32 [ undef, %entry ], [ %sub2, %while.body ]
  %c.0 = phi i32 [ undef, %entry ], [ %sub3, %while.body ]
  br i1 undef, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %sub = sub nsw i32 0, %b.0
  %sub2 = sub nsw i32 %sub, %c.0
  %sub3 = sub nsw i32 0, %c.0
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}
