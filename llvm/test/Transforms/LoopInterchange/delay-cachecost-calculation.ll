; REQUIRES: asserts

; RUN: opt -passes=loop-interchange -debug -disable-output %s 2>&1 | FileCheck %s

@A = global [16 x [16 x i32]] zeroinitializer

; Check that the CacheCost is calculated only when required. In this case, it
; is computed after passing the legality check.
;
; for (i = 0; i < 16; i++)
;   for (j = 0; j < 16; j++)
;     A[j][i] += 1;

; CHECK: Loops are legal to interchange
; CHECK: Compute CacheCost
define void @legal_to_interchange() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.next, %for.j ]
  %idx = getelementptr inbounds [16 x [16 x i32]], ptr @A, i32 0, i32 %j, i32 %i
  %val = load i32, ptr %idx
  %inc = add i32 %val, 1
  store i32 %inc, ptr %idx
  %j.next = add i32 %j, 1
  %j.exit = icmp eq i32 %j.next, 16
  br i1 %j.exit, label %for.i.latch, label %for.j

for.i.latch:
  %i.next = add i32 %i, 1
  %i.exit = icmp eq i32 %i.next, 16
  br i1 %i.exit, label %exit, label %for.i.header

exit:
  ret void
}

; Check that the CacheCost is not calculated when not required. In this case,
; the legality check always fails so that we do not need to compute the
; CacheCost.
;
; for (i = 0; i < 16; i++)
;   for (j = 0; j < 16; j++)
;     A[j][i] = A[i][j];

; CHECK-NOT: Compute CacheCost
define void @illegal_to_interchange() {
entry:
  br label %for.i.header

for.i.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.i.latch ]
  br label %for.j

for.j:
  %j = phi i32 [ 0, %for.i.header ], [ %j.next, %for.j ]
  %idx.load = getelementptr inbounds [16 x [16 x i32]], ptr @A, i32 0, i32 %i, i32 %j
  %idx.store = getelementptr inbounds [16 x [16 x i32]], ptr @A, i32 0, i32 %j, i32 %i
  %val = load i32, ptr %idx.load
  store i32 %val, ptr %idx.store
  %j.next = add i32 %j, 1
  %j.exit = icmp eq i32 %j.next, 16
  br i1 %j.exit, label %for.i.latch, label %for.j

for.i.latch:
  %i.next = add i32 %i, 1
  %i.exit = icmp eq i32 %i.next, 16
  br i1 %i.exit, label %exit, label %for.i.header

exit:
  ret void
}
