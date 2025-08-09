; RUN: opt < %s -passes=loop-interchange -pass-remarks-output=%t \
; RUN:     -disable-output
; RUN: FileCheck -input-file %t %s

; Check that loop interchange bail out early when all loops have dependencies
; in (potentially) all directions.
;
; for (int i = 0; i < 4; i++)
;   for (int j = 0; j < 4; j++)
;     A[i & val][j & val] = 42;

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            Dependence
; CHECK-NEXT: Function:        f
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          All loops have dependencies in all directions.
; CHECK-NEXT: ...
define void @f(ptr %A, i32 %val) {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.i.latch ]
  %subscript.0 = and i32 %i, %val
  %i2 = mul i32 %i, %i
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.next, %for.j ]
  %subscript.1 = and i32 %j, %val
  %idx = getelementptr inbounds [4 x [4 x i32]], ptr %A, i32 0, i32 %subscript.0, i32 %subscript.1
  store i32 42, ptr %idx, align 4
  %j.next = add i32 %j, 1
  %j.exit = icmp eq i32 %j.next, 4
  br i1 %j.exit, label %for.i.latch, label %for.j

for.i.latch:
  %i.next = add i32 %i, 1
  %i.exit = icmp eq i32 %i.next, 4
  br i1 %i.exit, label %exit, label %for.i.header

exit:
  ret void
}
