; REQUIRES: asserts
; RUN: opt < %s -disable-output -passes="print<da>" -debug-only=da 2>&1 | FileCheck %s

; for (int i = 0; i < n; i++)
;   a[i] = 0;
;
define void @single_loop_nsw(ptr %a, i64 %n) {
; CHECK-LABEL: 'single_loop_nsw'
; CHECK: Monotonicity: Monotonic expr: {0,+,1}<nuw><nsw><%loop>
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
define void @single_loop_nuw(ptr %a, i64 %begin, i64 %end) {
; CHECK-LABEL: 'single_loop_nuw'
; CHECK: Failed to prove monotonicity for: {%begin,+,1}<nuw><%loop>
; CHECK: Monotonicity: Unknown expr: {%begin,+,1}<nuw><%loop>
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
define void @nested_loop_nsw0(ptr %a, i64 %n, i64 %m) {
; CHECK-LABEL: 'nested_loop_nsw0'
; CHECK: Monotonicity: Monotonic expr: {{\{}}{0,+,1}<nuw><nsw><%loop.i.header>,+,1}<nuw><nsw><%loop.j>
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
;
define void @nested_loop_nsw1(ptr %a, i64 %n, i64 %m) {
; CHECK-LABEL: 'nested_loop_nsw1'
; CHECK: Monotonicity: Monotonic expr: {{\{}}{(-1 + %n),+,-1}<nsw><%loop.i.header>,+,1}<nsw><%loop.j>
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
;
define void @nested_loop_nuw(ptr %a, i64 %begin0, i64 %end0, i64 %begin1, i64 %end1) {
; CHECK-LABEL: 'nested_loop_nuw'
; CHECK: Failed to prove monotonicity for: {{\{}}{(%begin0 + %begin1),+,1}<nw><%loop.i.header>,+,1}<nw><%loop.j>
; CHECK: Monotonicity: Unknown expr: {{\{}}{(%begin0 + %begin1),+,1}<nw><%loop.i.header>,+,1}<nw><%loop.j>
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

; `step` can be zero, so `step*j` can be either Invariant or Monotonic. This
; propagates to `i + step*j`.
;
; for (int i = 0; i < n; i++)
;   for (int j = 0; j < m; j++)
;     a[i + step*j] = 0;
;
define void @nested_loop_step(ptr %a, i64 %n, i64 %m, i64 %step) {
; CHECK-LABEL: 'nested_loop_step'
; CHECK: Monotonicity: No signed wrap expr: {{\{}}{0,+,1}<nuw><nsw><%loop.i.header>,+,%step}<nsw><%loop.j>
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
  %offset.j = phi i64 [ 0, %loop.j.preheader ], [ %offset.j.next, %loop.j ]
  %offset = add nsw i64 %i, %offset.j
  %idx = getelementptr inbounds i8, ptr %a, i64 %offset
  store i8 0, ptr %idx
  %j.inc = add nsw i64 %j, 1
  %offset.j.next = add nsw i64 %offset.j, %step
  %exitcond.j = icmp eq i64 %j.inc, %m
  br i1 %exitcond.j, label %loop.i.latch, label %loop.j

loop.i.latch:
  %i.inc = add nsw i64 %i, 1
  %exitcond.i = icmp eq i64 %i.inc, %n
  br i1 %exitcond.i, label %exit, label %loop.i.header

exit:
  ret void
}

; `offset` can be loop-invariant since `step` can be zero.
;
; int8_t offset = start;
; for (int i = 0; i < 100; i++, offset += step)
;   a[sext(offset)] = 0;
;
define void @sext_nsw(ptr %a, i8 %start, i8 %step) {
; CHECK-LABEL: 'sext_nsw'
; CHECK: Monotonicity: No signed wrap expr: {(sext i8 %start to i64),+,(sext i8 %step to i64)}<nsw><%loop>
;
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %loop ]
  %offset = phi i8 [ %start, %entry ], [ %offset.next, %loop ]
  %offset.sext = sext i8 %offset to i64
  %idx = getelementptr i8, ptr %a, i64 %offset.sext
  store i8 0, ptr %idx
  %i.inc = add nsw i64 %i, 1
  %offset.next = add nsw i8 %offset, %step
  %exitcond = icmp eq i64 %i.inc, 100
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; The addition for `%offset.next` can wrap, so we cannot prove monotonicity.
;
; int8_t offset = start;
; for (int i = 0; i < 100; i++, offset += step)
;   a[sext(offset)] = 0;
;
define void @sext_may_wrap(ptr %a, i8 %start, i8 %step) {
; CHECK-LABEL: 'sext_may_wrap'
; CHECK: Failed to prove monotonicity for: {%start,+,%step}<%loop>
; CHECK: Monotonicity: Unknown expr: (sext i8 {%start,+,%step}<%loop> to i64)
;
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %loop ]
  %offset = phi i8 [ %start, %entry ], [ %offset.next, %loop ]
  %offset.sext = sext i8 %offset to i64
  %idx = getelementptr i8, ptr %a, i64 %offset.sext
  store i8 0, ptr %idx
  %i.inc = add nsw i64 %i, 1
  %offset.next = add i8 %offset, %step
  %exitcond = icmp eq i64 %i.inc, 100
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; `offset` can be loop-invariant since `step` can be zero.
;
; int8_t offset = start;
; for (int i = 0; i < 100; i++, offset += step)
;   a[zext(offset)] = 0;
;
define void @zext_nsw(ptr %a, i8 %start, i8 %step) {
; CHECK-LABEL: 'zext_nsw'
; CHECK: Monotonicity: No signed wrap expr: (zext i8 {%start,+,%step}<nsw><%loop> to i64)
;
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %loop ]
  %offset = phi i8 [ %start, %entry ], [ %offset.next, %loop ]
  %offset.zext = zext i8 %offset to i64
  %idx = getelementptr i8, ptr %a, i64 %offset.zext
  store i8 0, ptr %idx
  %i.inc = add nsw i64 %i, 1
  %offset.next = add nsw i8 %offset, %step
  %exitcond = icmp eq i64 %i.inc, 100
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; The addition for `%offset.next` can wrap, so we cannot prove monotonicity.
;
; int8_t offset = start;
; for (int i = 0; i < 100; i++, offset += step)
;   a[zext(offset)] = 0;
;
define void @zext_may_wrap(ptr %a, i8 %start, i8 %step) {
; CHECK-LABEL: 'zext_may_wrap'
; CHECK: Failed to prove monotonicity for: {%start,+,%step}<%loop>
; CHECK: Monotonicity: Unknown expr: (zext i8 {%start,+,%step}<%loop> to i64)
;
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %loop ]
  %offset = phi i8 [ %start, %entry ], [ %offset.next, %loop ]
  %offset.zext = zext i8 %offset to i64
  %idx = getelementptr i8, ptr %a, i64 %offset.zext
  store i8 0, ptr %idx
  %i.inc = add nsw i64 %i, 1
  %offset.next = add i8 %offset, %step
  %exitcond = icmp eq i64 %i.inc, 100
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
