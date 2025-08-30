; REQUIRES: asserts
; RUN: opt < %s -disable-output -passes="print<da>" -debug-only=da 2>&1 | FileCheck %s

; for (int i = 0; i < n; i++)
;   a[i] = 0;
;
define void @f0(ptr %a, i64 %n) {
; CHECK-LABEL: 'f0'
; CHECK: Monotonic expr: {0,+,1}<nuw><nsw><%loop>
;
entry:
  %guard = icmp sgt i64 %n, 0
  br i1 %guard, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %loop ]
  %idx = getelementptr inbounds i8, ptr %a, i64 %i
  store i8 0, ptr %idx
  %i.inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.inc, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; The purpose of the variable `begin` is to avoid violating the size limitation
; of the allocated object in LLVM IR, which would cause UB.
;
; for (unsigned long long i = begin; i < end; i++)
;   a[i] = 0;
;
define void @f1(ptr %a, i64 %begin, i64 %end) {
; CHECK-LABEL: 'f1'
; CHECK: Failed to prove monotonicity for: {%begin,+,1}<nuw><%loop>
;
entry:
  %guard = icmp ult i64 %begin, %end
  br i1 %guard, label %loop, label %exit

loop:
  %i = phi i64 [ %begin, %entry ], [ %i.inc, %loop ]
  %idx = getelementptr i8, ptr %a, i64 %i
  store i8 0, ptr %idx
  %i.inc = add nuw i64 %i, 1
  %exitcond = icmp eq i64 %i.inc, %end
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; for (int i = 0; i < n; i++)
;   for (int j = 0; j < m; j++)
;     a[i + j] = 0;
;
define void @f2(ptr %a, i64 %n, i64 %m) {
; CHECK-LABEL: 'f2'
; CHECK: Monotonic expr: {{\{}}{0,+,1}<nuw><nsw><%loop.i.header>,+,1}<nuw><nsw><%loop.j>
;
entry:
  %guard.i = icmp sgt i64 %n, 0
  br i1 %guard.i, label %loop.i.header, label %exit

loop.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %loop.i.latch ]
  br label %loop.j.preheader

loop.j.preheader:
  %gurard.j = icmp sgt i64 %m, 0
  br i1 %gurard.j, label %loop.j, label %loop.i.latch

loop.j:
  %j = phi i64 [ 0, %loop.j.preheader ], [ %j.inc, %loop.j ]
  %offset = add nsw i64 %i, %j
  %idx = getelementptr inbounds i8, ptr %a, i64 %offset
  store i8 0, ptr %idx
  %j.inc = add nsw i64 %j, 1
  %exitcond.j = icmp eq i64 %j.inc, %m
  br i1 %exitcond.j, label %loop.i.latch, label %loop.j

loop.i.latch:
  %i.inc = add nsw i64 %i, 1
  %exitcond.i = icmp eq i64 %i.inc, %n
  br i1 %exitcond.i, label %exit, label %loop.i.header

exit:
  ret void
}

; for (int i = n - 1; i >= 0; i--)
;   for (int j = 0; j < m; j++)
;     a[i + j] = 0;
define void @f3(ptr %a, i64 %n, i64 %m) {
; CHECK-LABEL: 'f3'
; CHECK: Monotonic expr: {{\{}}{(-1 + %n),+,-1}<nsw><%loop.i.header>,+,1}<nsw><%loop.j>
;
entry:
  %guard.i = icmp sgt i64 %n, 0
  br i1 %guard.i, label %loop.i.header, label %exit

loop.i.header:
  %i = phi i64 [ %n, %entry ], [ %i.dec, %loop.i.latch ]
  %i.dec = add nsw i64 %i, -1
  br label %loop.j.preheader

loop.j.preheader:
  %gurard.j = icmp sgt i64 %m, 0
  br i1 %gurard.j, label %loop.j, label %loop.i.latch

loop.j:
  %j = phi i64 [ 0, %loop.j.preheader ], [ %j.inc, %loop.j ]
  %offset = add nsw i64 %i.dec, %j
  %idx = getelementptr inbounds i8, ptr %a, i64 %offset
  store i8 0, ptr %idx
  %j.inc = add nsw i64 %j, 1
  %exitcond.j = icmp eq i64 %j.inc, %m
  br i1 %exitcond.j, label %loop.i.latch, label %loop.j

loop.i.latch:
  %exitcond.i = icmp eq i64 %i.dec, 0
  br i1 %exitcond.i, label %exit, label %loop.i.header

exit:
  ret void
}

; for (int i = begin0; i < end0; i++)
;   for (int j = begin1; j < end1; j++) {
;     unsigned long long offset = (unsigned long long)i + (unsigned long long)j;
;     a[offset] = 0;
;   }
define void @f4(ptr %a, i64 %begin0, i64 %end0, i64 %begin1, i64 %end1) {
; CHECK-LABEL: 'f4'
; CHECK: Failed to prove monotonicity for: {{\{}}{(%begin0 + %begin1),+,1}<nw><%loop.i.header>,+,1}<nw><%loop.j>
;
entry:
  %guard.i.0 = icmp slt i64 0, %begin0
  %guard.i.1 = icmp slt i64 %begin0, %end0
  %guard.i.2 = icmp slt i64 0, %end0
  %and.i.0 = and i1 %guard.i.0, %guard.i.1
  %and.i.1 = and i1 %and.i.0, %guard.i.2
  br i1 %and.i.1, label %loop.i.header, label %exit

loop.i.header:
  %i = phi i64 [ %begin0, %entry ], [ %i.inc, %loop.i.latch ]
  br label %loop.j.preheader

loop.j.preheader:
  %guard.j.0 = icmp slt i64 0, %begin1
  %guard.j.1 = icmp slt i64 %begin1, %end1
  %guard.j.2 = icmp slt i64 0, %end1
  %and.j.0 = and i1 %guard.j.0, %guard.j.1
  %and.j.1 = and i1 %and.j.0, %guard.j.2
  br i1 %and.j.1, label %loop.j, label %loop.i.latch

loop.j:
  %j = phi i64 [ %begin1, %loop.j.preheader ], [ %j.inc, %loop.j ]
  %offset = add nuw i64 %i, %j
  %idx = getelementptr i8, ptr %a, i64 %offset
  store i8 0, ptr %idx
  %j.inc = add nsw i64 %j, 1
  %exitcond.j = icmp eq i64 %j.inc, %end1
  br i1 %exitcond.j, label %loop.i.latch, label %loop.j

loop.i.latch:
  %i.inc = add nsw i64 %i, 1
  %exitcond.i = icmp eq i64 %i.inc, %end0
  br i1 %exitcond.i, label %exit, label %loop.i.header

exit:
  ret void
}
