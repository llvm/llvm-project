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
