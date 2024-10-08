; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s

%i8.2 = type {i8, i8}
define void @i8_factor_2(ptr %data, i64 %n) {
entry:
  br label %for.body
; CHECK-LABEL: Checking a loop in 'i8_factor_2'
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 3 for VF 16: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 3 for VF 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 5 for VF 32: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 5 for VF 32: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 2 for VF vscale x 1: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF vscale x 1: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 2 for VF vscale x 2: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF vscale x 2: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 2 for VF vscale x 4: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF vscale x 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 3 for VF vscale x 8: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 3 for VF vscale x 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; CHECK: Cost of 5 for VF vscale x 16: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; CHECK: Cost of 5 for VF vscale x 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %p0 = getelementptr inbounds %i8.2, ptr %data, i64 %i, i32 0
  %p1 = getelementptr inbounds %i8.2, ptr %data, i64 %i, i32 1
  %l0 = load i8, ptr %p0, align 1
  %l1 = load i8, ptr %p1, align 1
  %a0 = add i8 %l0, 1
  %a1 = add i8 %l1, 2
  store i8 %a0, ptr %p0, align 1
  store i8 %a1, ptr %p1, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i8.3 = type {i8, i8, i8}
define void @i8_factor_3(ptr %data, i64 %n) {
entry:
  br label %for.body
; CHECK-LABEL: Checking a loop in 'i8_factor_3'
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; CHECK: Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; CHECK: Cost of 3 for VF 8: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; CHECK: Cost of 3 for VF 8: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; CHECK: Cost of 5 for VF 16: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; CHECK: Cost of 5 for VF 16: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; CHECK: Cost of 9 for VF 32: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; CHECK: Cost of 9 for VF 32: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %p0 = getelementptr inbounds %i8.3, ptr %data, i64 %i, i32 0
  %p1 = getelementptr inbounds %i8.3, ptr %data, i64 %i, i32 1
  %p2 = getelementptr inbounds %i8.3, ptr %data, i64 %i, i32 2
  %l0 = load i8, ptr %p0, align 1
  %l1 = load i8, ptr %p1, align 1
  %l2 = load i8, ptr %p2, align 1
  %a0 = add i8 %l0, 1
  %a1 = add i8 %l1, 2
  %a2 = add i8 %l2, 3
  store i8 %a0, ptr %p0, align 1
  store i8 %a1, ptr %p1, align 1
  store i8 %a2, ptr %p2, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i8.4 = type {i8, i8, i8, i8}
define void @i8_factor_4(ptr %data, i64 %n) {
entry:
  br label %for.body
; CHECK-LABEL: Checking a loop in 'i8_factor_4'
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; CHECK: Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; CHECK: Cost of 3 for VF 8: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; CHECK: Cost of 3 for VF 8: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; CHECK: Cost of 5 for VF 16: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; CHECK: Cost of 5 for VF 16: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; CHECK: Cost of 9 for VF 32: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; CHECK: Cost of 9 for VF 32: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %p0 = getelementptr inbounds %i8.4, ptr %data, i64 %i, i32 0
  %p1 = getelementptr inbounds %i8.4, ptr %data, i64 %i, i32 1
  %p2 = getelementptr inbounds %i8.4, ptr %data, i64 %i, i32 2
  %p3 = getelementptr inbounds %i8.4, ptr %data, i64 %i, i32 3
  %l0 = load i8, ptr %p0, align 1
  %l1 = load i8, ptr %p1, align 1
  %l2 = load i8, ptr %p2, align 1
  %l3 = load i8, ptr %p3, align 1
  %a0 = add i8 %l0, 1
  %a1 = add i8 %l1, 2
  %a2 = add i8 %l2, 3
  %a3 = add i8 %l3, 4
  store i8 %a0, ptr %p0, align 1
  store i8 %a1, ptr %p1, align 1
  store i8 %a2, ptr %p2, align 1
  store i8 %a3, ptr %p3, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i8.5 = type {i8, i8, i8, i8, i8}
define void @i8_factor_5(ptr %data, i64 %n) {
entry:
  br label %for.body
; CHECK-LABEL: Checking a loop in 'i8_factor_5'
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; CHECK: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; CHECK: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; CHECK: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; CHECK: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; CHECK: Cost of 9 for VF 16: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; CHECK: Cost of 9 for VF 16: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %p0 = getelementptr inbounds %i8.5, ptr %data, i64 %i, i32 0
  %p1 = getelementptr inbounds %i8.5, ptr %data, i64 %i, i32 1
  %p2 = getelementptr inbounds %i8.5, ptr %data, i64 %i, i32 2
  %p3 = getelementptr inbounds %i8.5, ptr %data, i64 %i, i32 3
  %p4 = getelementptr inbounds %i8.5, ptr %data, i64 %i, i32 4
  %l0 = load i8, ptr %p0, align 1
  %l1 = load i8, ptr %p1, align 1
  %l2 = load i8, ptr %p2, align 1
  %l3 = load i8, ptr %p3, align 1
  %l4 = load i8, ptr %p4, align 1
  %a0 = add i8 %l0, 1
  %a1 = add i8 %l1, 2
  %a2 = add i8 %l2, 3
  %a3 = add i8 %l3, 4
  %a4 = add i8 %l4, 5
  store i8 %a0, ptr %p0, align 1
  store i8 %a1, ptr %p1, align 1
  store i8 %a2, ptr %p2, align 1
  store i8 %a3, ptr %p3, align 1
  store i8 %a4, ptr %p4, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i8.6 = type {i8, i8, i8, i8, i8, i8}
define void @i8_factor_6(ptr %data, i64 %n) {
entry:
  br label %for.body
; CHECK-LABEL: Checking a loop in 'i8_factor_6'
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; CHECK: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; CHECK: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; CHECK: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; CHECK: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; CHECK: Cost of 9 for VF 16: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; CHECK: Cost of 9 for VF 16: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %p0 = getelementptr inbounds %i8.6, ptr %data, i64 %i, i32 0
  %p1 = getelementptr inbounds %i8.6, ptr %data, i64 %i, i32 1
  %p2 = getelementptr inbounds %i8.6, ptr %data, i64 %i, i32 2
  %p3 = getelementptr inbounds %i8.6, ptr %data, i64 %i, i32 3
  %p4 = getelementptr inbounds %i8.6, ptr %data, i64 %i, i32 4
  %p5 = getelementptr inbounds %i8.6, ptr %data, i64 %i, i32 5
  %l0 = load i8, ptr %p0, align 1
  %l1 = load i8, ptr %p1, align 1
  %l2 = load i8, ptr %p2, align 1
  %l3 = load i8, ptr %p3, align 1
  %l4 = load i8, ptr %p4, align 1
  %l5 = load i8, ptr %p5, align 1
  %a0 = add i8 %l0, 1
  %a1 = add i8 %l1, 2
  %a2 = add i8 %l2, 3
  %a3 = add i8 %l3, 4
  %a4 = add i8 %l4, 5
  %a5 = add i8 %l5, 6
  store i8 %a0, ptr %p0, align 1
  store i8 %a1, ptr %p1, align 1
  store i8 %a2, ptr %p2, align 1
  store i8 %a3, ptr %p3, align 1
  store i8 %a4, ptr %p4, align 1
  store i8 %a5, ptr %p5, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i8.7 = type {i8, i8, i8, i8, i8, i8, i8}
define void @i8_factor_7(ptr %data, i64 %n) {
entry:
  br label %for.body
; CHECK-LABEL: Checking a loop in 'i8_factor_7'
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; CHECK: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; CHECK: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; CHECK: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; CHECK: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; CHECK: Cost of 9 for VF 16: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; CHECK: Cost of 9 for VF 16: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %p0 = getelementptr inbounds %i8.7, ptr %data, i64 %i, i32 0
  %p1 = getelementptr inbounds %i8.7, ptr %data, i64 %i, i32 1
  %p2 = getelementptr inbounds %i8.7, ptr %data, i64 %i, i32 2
  %p3 = getelementptr inbounds %i8.7, ptr %data, i64 %i, i32 3
  %p4 = getelementptr inbounds %i8.7, ptr %data, i64 %i, i32 4
  %p5 = getelementptr inbounds %i8.7, ptr %data, i64 %i, i32 5
  %p6 = getelementptr inbounds %i8.7, ptr %data, i64 %i, i32 6
  %l0 = load i8, ptr %p0, align 1
  %l1 = load i8, ptr %p1, align 1
  %l2 = load i8, ptr %p2, align 1
  %l3 = load i8, ptr %p3, align 1
  %l4 = load i8, ptr %p4, align 1
  %l5 = load i8, ptr %p5, align 1
  %l6 = load i8, ptr %p6, align 1
  %a0 = add i8 %l0, 1
  %a1 = add i8 %l1, 2
  %a2 = add i8 %l2, 3
  %a3 = add i8 %l3, 4
  %a4 = add i8 %l4, 5
  %a5 = add i8 %l5, 6
  %a6 = add i8 %l6, 7
  store i8 %a0, ptr %p0, align 1
  store i8 %a1, ptr %p1, align 1
  store i8 %a2, ptr %p2, align 1
  store i8 %a3, ptr %p3, align 1
  store i8 %a4, ptr %p4, align 1
  store i8 %a5, ptr %p5, align 1
  store i8 %a6, ptr %p6, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i8.8 = type {i8, i8, i8, i8, i8, i8, i8, i8}
define void @i8_factor_8(ptr %data, i64 %n) {
entry:
  br label %for.body
; CHECK-LABEL: Checking a loop in 'i8_factor_8'
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; CHECK: Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; CHECK: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; CHECK: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; CHECK: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; CHECK: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; CHECK: Cost of 9 for VF 16: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; CHECK: Cost of 9 for VF 16: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %p0 = getelementptr inbounds %i8.8, ptr %data, i64 %i, i32 0
  %p1 = getelementptr inbounds %i8.8, ptr %data, i64 %i, i32 1
  %p2 = getelementptr inbounds %i8.8, ptr %data, i64 %i, i32 2
  %p3 = getelementptr inbounds %i8.8, ptr %data, i64 %i, i32 3
  %p4 = getelementptr inbounds %i8.8, ptr %data, i64 %i, i32 4
  %p5 = getelementptr inbounds %i8.8, ptr %data, i64 %i, i32 5
  %p6 = getelementptr inbounds %i8.8, ptr %data, i64 %i, i32 6
  %p7 = getelementptr inbounds %i8.8, ptr %data, i64 %i, i32 7
  %l0 = load i8, ptr %p0, align 1
  %l1 = load i8, ptr %p1, align 1
  %l2 = load i8, ptr %p2, align 1
  %l3 = load i8, ptr %p3, align 1
  %l4 = load i8, ptr %p4, align 1
  %l5 = load i8, ptr %p5, align 1
  %l6 = load i8, ptr %p6, align 1
  %l7 = load i8, ptr %p7, align 1
  %a0 = add i8 %l0, 1
  %a1 = add i8 %l1, 2
  %a2 = add i8 %l2, 3
  %a3 = add i8 %l3, 4
  %a4 = add i8 %l4, 5
  %a5 = add i8 %l5, 6
  %a6 = add i8 %l6, 7
  %a7 = add i8 %l7, 8
  store i8 %a0, ptr %p0, align 1
  store i8 %a1, ptr %p1, align 1
  store i8 %a2, ptr %p2, align 1
  store i8 %a3, ptr %p3, align 1
  store i8 %a4, ptr %p4, align 1
  store i8 %a5, ptr %p5, align 1
  store i8 %a6, ptr %p6, align 1
  store i8 %a7, ptr %p7, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
