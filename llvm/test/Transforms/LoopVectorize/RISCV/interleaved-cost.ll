; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v,-optimized-nf2-segment-load-store -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NO-OPT
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=OPT-NF2
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v,+optimized-nf3-segment-load-store -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=OPT-NF3
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v,+optimized-nf4-segment-load-store -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=OPT-NF4
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v,+optimized-nf5-segment-load-store -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=OPT-NF5
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v,+optimized-nf6-segment-load-store -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=OPT-NF6
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v,+optimized-nf7-segment-load-store -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=OPT-NF7
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v,+optimized-nf8-segment-load-store -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=OPT-NF8

%i8.2 = type {i8, i8}
define void @i8_factor_2(ptr %data, i64 %n) {
entry:
  br label %for.body
; OPT-NF2-LABEL: Checking a loop in 'i8_factor_2'
; OPT-NF2: Cost of 3 for VF 2: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 3 for VF 2: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 3 for VF 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 3 for VF 8: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 3 for VF 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 4 for VF 16: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 4 for VF 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 8 for VF 32: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 8 for VF 32: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 3 for VF vscale x 1: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 3 for VF vscale x 1: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 3 for VF vscale x 2: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 3 for VF vscale x 2: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 3 for VF vscale x 4: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 3 for VF vscale x 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 4 for VF vscale x 8: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 4 for VF vscale x 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; OPT-NF2: Cost of 8 for VF vscale x 16: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; OPT-NF2: Cost of 8 for VF vscale x 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT-LABEL: Checking a loop in 'i8_factor_2'
; NO-OPT: Cost of 4 for VF 2: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 4 for VF 2: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 8 for VF 4: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 8 for VF 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 16 for VF 8: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 16 for VF 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 32 for VF 16: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 32 for VF 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 64 for VF 32: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 64 for VF 32: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 4 for VF vscale x 1: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 4 for VF vscale x 1: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 8 for VF vscale x 2: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 8 for VF vscale x 2: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 16 for VF vscale x 4: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 16 for VF vscale x 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 32 for VF vscale x 8: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 32 for VF vscale x 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
; NO-OPT: Cost of 64 for VF vscale x 16: INTERLEAVE-GROUP with factor 2 at %l0, ir<%p0>
; NO-OPT: Cost of 64 for VF vscale x 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%p0>
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
; OPT-NF3-LABEL: Checking a loop in 'i8_factor_3'
; OPT-NF3: Cost of 4 for VF 2: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 4 for VF 2: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 4 for VF 4: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 4 for VF 4: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 5 for VF 8: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 7 for VF 16: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 7 for VF 16: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 14 for VF 32: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 14 for VF 32: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 4 for VF vscale x 1: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 4 for VF vscale x 1: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 4 for VF vscale x 2: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 4 for VF vscale x 2: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 5 for VF vscale x 4: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 5 for VF vscale x 4: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 7 for VF vscale x 8: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 7 for VF vscale x 8: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; OPT-NF3: Cost of 14 for VF vscale x 16: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; OPT-NF3: Cost of 14 for VF vscale x 16: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT-LABEL: Checking a loop in 'i8_factor_3'
; NO-OPT: Cost of 6 for VF 2: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 6 for VF 2: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 12 for VF 4: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 12 for VF 4: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 24 for VF 8: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 24 for VF 8: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 48 for VF 16: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 48 for VF 16: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 96 for VF 32: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 96 for VF 32: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 6 for VF vscale x 1: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 6 for VF vscale x 1: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 12 for VF vscale x 2: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 12 for VF vscale x 2: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 24 for VF vscale x 4: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 24 for VF vscale x 4: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 48 for VF vscale x 8: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 48 for VF vscale x 8: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
; NO-OPT: Cost of 96 for VF vscale x 16: INTERLEAVE-GROUP with factor 3 at %l0, ir<%p0>
; NO-OPT: Cost of 96 for VF vscale x 16: INTERLEAVE-GROUP with factor 3 at <badref>, ir<%p0>
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
; OPT-NF4-LABEL: Checking a loop in 'i8_factor_4'
; OPT-NF4: Cost of 5 for VF 2: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 5 for VF 2: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 5 for VF 4: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 5 for VF 4: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 6 for VF 8: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 6 for VF 8: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 8 for VF 16: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 8 for VF 16: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 16 for VF 32: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 16 for VF 32: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 5 for VF vscale x 1: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 5 for VF vscale x 1: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 5 for VF vscale x 2: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 5 for VF vscale x 2: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 6 for VF vscale x 4: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 6 for VF vscale x 4: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 8 for VF vscale x 8: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 8 for VF vscale x 8: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; OPT-NF4: Cost of 16 for VF vscale x 16: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; OPT-NF4: Cost of 16 for VF vscale x 16: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT-LABEL: Checking a loop in 'i8_factor_4'
; NO-OPT: Cost of 8 for VF 2: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 8 for VF 2: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 16 for VF 4: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 16 for VF 4: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 32 for VF 8: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 32 for VF 8: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 64 for VF 16: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 64 for VF 16: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 128 for VF 32: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 128 for VF 32: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 8 for VF vscale x 1: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 8 for VF vscale x 1: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 16 for VF vscale x 2: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 16 for VF vscale x 2: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 32 for VF vscale x 4: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 32 for VF vscale x 4: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 64 for VF vscale x 8: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 64 for VF vscale x 8: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
; NO-OPT: Cost of 128 for VF vscale x 16: INTERLEAVE-GROUP with factor 4 at %l0, ir<%p0>
; NO-OPT: Cost of 128 for VF vscale x 16: INTERLEAVE-GROUP with factor 4 at <badref>, ir<%p0>
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
; OPT-NF5-LABEL: Checking a loop in 'i8_factor_5'
; OPT-NF5: Cost of 6 for VF 2: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; OPT-NF5: Cost of 6 for VF 2: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; OPT-NF5: Cost of 7 for VF 4: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; OPT-NF5: Cost of 7 for VF 4: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; OPT-NF5: Cost of 9 for VF 8: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; OPT-NF5: Cost of 9 for VF 8: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; OPT-NF5: Cost of 13 for VF 16: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; OPT-NF5: Cost of 13 for VF 16: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; OPT-NF5: Cost of 6 for VF vscale x 1: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; OPT-NF5: Cost of 6 for VF vscale x 1: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; OPT-NF5: Cost of 7 for VF vscale x 2: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; OPT-NF5: Cost of 7 for VF vscale x 2: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; OPT-NF5: Cost of 9 for VF vscale x 4: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; OPT-NF5: Cost of 9 for VF vscale x 4: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; OPT-NF5: Cost of 13 for VF vscale x 8: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; OPT-NF5: Cost of 13 for VF vscale x 8: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; NO-OPT-LABEL: Checking a loop in 'i8_factor_5'
; NO-OPT: Cost of 10 for VF 2: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; NO-OPT: Cost of 10 for VF 2: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; NO-OPT: Cost of 20 for VF 4: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; NO-OPT: Cost of 20 for VF 4: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; NO-OPT: Cost of 40 for VF 8: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; NO-OPT: Cost of 40 for VF 8: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; NO-OPT: Cost of 80 for VF 16: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; NO-OPT: Cost of 80 for VF 16: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; NO-OPT: Cost of 10 for VF vscale x 1: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; NO-OPT: Cost of 10 for VF vscale x 1: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; NO-OPT: Cost of 20 for VF vscale x 2: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; NO-OPT: Cost of 20 for VF vscale x 2: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; NO-OPT: Cost of 40 for VF vscale x 4: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; NO-OPT: Cost of 40 for VF vscale x 4: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
; NO-OPT: Cost of 80 for VF vscale x 8: INTERLEAVE-GROUP with factor 5 at %l0, ir<%p0>
; NO-OPT: Cost of 80 for VF vscale x 8: INTERLEAVE-GROUP with factor 5 at <badref>, ir<%p0>
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
; OPT-NF6-LABEL: Checking a loop in 'i8_factor_6'
; OPT-NF6: Cost of 7 for VF 2: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; OPT-NF6: Cost of 7 for VF 2: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; OPT-NF6: Cost of 8 for VF 4: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; OPT-NF6: Cost of 8 for VF 4: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; OPT-NF6: Cost of 10 for VF 8: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; OPT-NF6: Cost of 10 for VF 8: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; OPT-NF6: Cost of 14 for VF 16: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; OPT-NF6: Cost of 14 for VF 16: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; OPT-NF6: Cost of 7 for VF vscale x 1: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; OPT-NF6: Cost of 7 for VF vscale x 1: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; OPT-NF6: Cost of 8 for VF vscale x 2: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; OPT-NF6: Cost of 8 for VF vscale x 2: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; OPT-NF6: Cost of 10 for VF vscale x 4: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; OPT-NF6: Cost of 10 for VF vscale x 4: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; OPT-NF6: Cost of 14 for VF vscale x 8: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; OPT-NF6: Cost of 14 for VF vscale x 8: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; NO-OPT-LABEL: Checking a loop in 'i8_factor_6'
; NO-OPT: Cost of 12 for VF 2: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; NO-OPT: Cost of 12 for VF 2: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; NO-OPT: Cost of 24 for VF 4: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; NO-OPT: Cost of 24 for VF 4: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; NO-OPT: Cost of 48 for VF 8: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; NO-OPT: Cost of 48 for VF 8: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; NO-OPT: Cost of 96 for VF 16: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; NO-OPT: Cost of 96 for VF 16: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; NO-OPT: Cost of 12 for VF vscale x 1: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; NO-OPT: Cost of 12 for VF vscale x 1: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; NO-OPT: Cost of 24 for VF vscale x 2: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; NO-OPT: Cost of 24 for VF vscale x 2: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; NO-OPT: Cost of 48 for VF vscale x 4: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; NO-OPT: Cost of 48 for VF vscale x 4: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
; NO-OPT: Cost of 96 for VF vscale x 8: INTERLEAVE-GROUP with factor 6 at %l0, ir<%p0>
; NO-OPT: Cost of 96 for VF vscale x 8: INTERLEAVE-GROUP with factor 6 at <badref>, ir<%p0>
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
; OPT-NF7-LABEL: Checking a loop in 'i8_factor_7'
; OPT-NF7: Cost of 8 for VF 2: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; OPT-NF7: Cost of 8 for VF 2: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; OPT-NF7: Cost of 9 for VF 4: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; OPT-NF7: Cost of 9 for VF 4: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; OPT-NF7: Cost of 11 for VF 8: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; OPT-NF7: Cost of 11 for VF 8: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; OPT-NF7: Cost of 15 for VF 16: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; OPT-NF7: Cost of 15 for VF 16: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; OPT-NF7: Cost of 8 for VF vscale x 1: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; OPT-NF7: Cost of 8 for VF vscale x 1: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; OPT-NF7: Cost of 9 for VF vscale x 2: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; OPT-NF7: Cost of 9 for VF vscale x 2: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; OPT-NF7: Cost of 11 for VF vscale x 4: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; OPT-NF7: Cost of 11 for VF vscale x 4: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; OPT-NF7: Cost of 15 for VF vscale x 8: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; OPT-NF7: Cost of 15 for VF vscale x 8: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; NO-OPT-LABEL: Checking a loop in 'i8_factor_7'
; NO-OPT: Cost of 14 for VF 2: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; NO-OPT: Cost of 14 for VF 2: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; NO-OPT: Cost of 28 for VF 4: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; NO-OPT: Cost of 28 for VF 4: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; NO-OPT: Cost of 56 for VF 8: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; NO-OPT: Cost of 56 for VF 8: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; NO-OPT: Cost of 112 for VF 16: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; NO-OPT: Cost of 112 for VF 16: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; NO-OPT: Cost of 14 for VF vscale x 1: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; NO-OPT: Cost of 14 for VF vscale x 1: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; NO-OPT: Cost of 28 for VF vscale x 2: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; NO-OPT: Cost of 28 for VF vscale x 2: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; NO-OPT: Cost of 56 for VF vscale x 4: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; NO-OPT: Cost of 56 for VF vscale x 4: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
; NO-OPT: Cost of 112 for VF vscale x 8: INTERLEAVE-GROUP with factor 7 at %l0, ir<%p0>
; NO-OPT: Cost of 112 for VF vscale x 8: INTERLEAVE-GROUP with factor 7 at <badref>, ir<%p0>
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
; OPT-NF8-LABEL: Checking a loop in 'i8_factor_8'
; OPT-NF8: Cost of 9 for VF 2: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; OPT-NF8: Cost of 9 for VF 2: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; OPT-NF8: Cost of 10 for VF 4: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; OPT-NF8: Cost of 10 for VF 4: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; OPT-NF8: Cost of 12 for VF 8: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; OPT-NF8: Cost of 12 for VF 8: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; OPT-NF8: Cost of 16 for VF 16: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; OPT-NF8: Cost of 16 for VF 16: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; OPT-NF8: Cost of 9 for VF vscale x 1: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; OPT-NF8: Cost of 9 for VF vscale x 1: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; OPT-NF8: Cost of 10 for VF vscale x 2: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; OPT-NF8: Cost of 10 for VF vscale x 2: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; OPT-NF8: Cost of 12 for VF vscale x 4: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; OPT-NF8: Cost of 12 for VF vscale x 4: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; OPT-NF8: Cost of 16 for VF vscale x 8: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; OPT-NF8: Cost of 16 for VF vscale x 8: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; NO-OPT-LABEL: Checking a loop in 'i8_factor_8'
; NO-OPT: Cost of 16 for VF 2: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; NO-OPT: Cost of 16 for VF 2: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; NO-OPT: Cost of 32 for VF 4: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; NO-OPT: Cost of 32 for VF 4: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; NO-OPT: Cost of 64 for VF 8: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; NO-OPT: Cost of 64 for VF 8: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; NO-OPT: Cost of 128 for VF 16: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; NO-OPT: Cost of 128 for VF 16: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; NO-OPT: Cost of 16 for VF vscale x 1: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; NO-OPT: Cost of 16 for VF vscale x 1: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; NO-OPT: Cost of 32 for VF vscale x 2: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; NO-OPT: Cost of 32 for VF vscale x 2: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; NO-OPT: Cost of 64 for VF vscale x 4: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; NO-OPT: Cost of 64 for VF vscale x 4: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
; NO-OPT: Cost of 128 for VF vscale x 8: INTERLEAVE-GROUP with factor 8 at %l0, ir<%p0>
; NO-OPT: Cost of 128 for VF vscale x 8: INTERLEAVE-GROUP with factor 8 at <badref>, ir<%p0>
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
