; RUN: opt < %s -passes=loop-interchange -pass-remarks-output=%t \
; RUN:     -disable-output
; RUN: FileCheck -input-file %t %s

; Check that loop interchange bails out early when finding a direction vector
; with all '*' elements.
;
; for (int i = 0; i < 4; i++)
;   for (int j = 0; j < 4; j++)
;     A[i & val][j & val] = 0;

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        f
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          All loops have dependencies in all directions.
; CHECK-NEXT: ...
define void @f(ptr %A, i64 %val) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i64 [ 0, %for.i.header ], [ %j.next, %for.j ]
  %subscript.0 = and i64 %i, %val
  %subscript.1 = and i64 %j, %val
  %idx = getelementptr inbounds [4 x i8], ptr %A, i64 %subscript.0, i64 %subscript.1
  store i8 0, ptr %idx
  %j.next = add nuw nsw i64 %j, 1
  %exit.j = icmp eq i64 %j.next, 4
  br i1 %exit.j, label %for.i.latch, label %for.j

for.i.latch:
  %i.next = add nuw nsw i64 %i, 1
  %exit.i = icmp eq i64 %i.next, 4
  br i1 %exit.i, label %exit, label %for.i.header

exit:
  ret void
}
