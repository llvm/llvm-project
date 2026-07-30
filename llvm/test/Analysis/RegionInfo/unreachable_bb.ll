; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

; We should not crash if there are some bbs that are not reachable.
define void @f() {
entry:
  br label %for.pre

notintree:                                        ; No predecessors!
  br label %ret

for.pre:                                          ; preds = %entry
  br label %for

for:                                              ; preds = %for.inc, %for.pre
  %indvar = phi i64 [ 0, %for.pre ], [ %indvar.next, %for.inc ]
  %exitcond = icmp ne i64 %indvar, 200
  br i1 %exitcond, label %for.inc, label %ret

for.inc:                                          ; preds = %for
  %indvar.next = add i64 %indvar, 1
  br label %for

ret:                                              ; preds = %for, %notintree
  ret void
}

; CHECK: [0] entry => <Function Return>
; CHECK:   [1] for => ret

