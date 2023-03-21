; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -force-vector-width=2 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_2
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -force-vector-width=4 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_4
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -force-vector-width=8 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_8
; RUN: opt -passes=loop-vectorize -mtriple=riscv64 -mattr=+v -force-vector-width=16 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_16

%i8.2 = type {i8, i8}
define void @i8_factor_2(ptr %data, i64 %n) {
entry:
  br label %for.body
; VF_2-LABEL: Checking a loop in 'i8_factor_2'
; VF_2:       Found an estimated cost of 2 for VF 2 For instruction:   %l0 = load i8, ptr %p0, align 1
; VF_2-NEXT:  Found an estimated cost of 0 for VF 2 For instruction:   %l1 = load i8, ptr %p1, align 1
; VF_2:       Found an estimated cost of 0 for VF 2 For instruction:   store i8 %a0, ptr %p0, align 1
; VF_2-NEXT:  Found an estimated cost of 2 for VF 2 For instruction:   store i8 %a1, ptr %p1, align 1
; VF_4-LABEL: Checking a loop in 'i8_factor_2'
; VF_4:       Found an estimated cost of 2 for VF 4 For instruction:   %l0 = load i8, ptr %p0, align 1
; VF_4-NEXT:  Found an estimated cost of 0 for VF 4 For instruction:   %l1 = load i8, ptr %p1, align 1
; VF_4:       Found an estimated cost of 0 for VF 4 For instruction:   store i8 %a0, ptr %p0, align 1
; VF_4-NEXT:  Found an estimated cost of 2 for VF 4 For instruction:   store i8 %a1, ptr %p1, align 1
; VF_8-LABEL: Checking a loop in 'i8_factor_2'
; VF_8:       Found an estimated cost of 2 for VF 8 For instruction:   %l0 = load i8, ptr %p0, align 1
; VF_8-NEXT:  Found an estimated cost of 0 for VF 8 For instruction:   %l1 = load i8, ptr %p1, align 1
; VF_8:       Found an estimated cost of 0 for VF 8 For instruction:   store i8 %a0, ptr %p0, align 1
; VF_8-NEXT:  Found an estimated cost of 2 for VF 8 For instruction:   store i8 %a1, ptr %p1, align 1
; VF_16-LABEL: Checking a loop in 'i8_factor_2'
; VF_16:       Found an estimated cost of 2 for VF 16 For instruction:   %l0 = load i8, ptr %p0, align 1
; VF_16-NEXT:  Found an estimated cost of 0 for VF 16 For instruction:   %l1 = load i8, ptr %p1, align 1
; VF_16:       Found an estimated cost of 0 for VF 16 For instruction:   store i8 %a0, ptr %p0, align 1
; VF_16-NEXT:  Found an estimated cost of 2 for VF 16 For instruction:   store i8 %a1, ptr %p1, align 1
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
; VF_2-LABEL: Checking a loop in 'i8_factor_3'
; VF_2:       Found an estimated cost of 2 for VF 2 For instruction:   %l0 = load i8, ptr %p0, align 1
; VF_2-NEXT:  Found an estimated cost of 0 for VF 2 For instruction:   %l1 = load i8, ptr %p1, align 1
; VF_2-NEXT:  Found an estimated cost of 0 for VF 2 For instruction:   %l2 = load i8, ptr %p2, align 1
; VF_2:       Found an estimated cost of 0 for VF 2 For instruction:   store i8 %a0, ptr %p0, align 1
; VF_2:       Found an estimated cost of 0 for VF 2 For instruction:   store i8 %a1, ptr %p1, align 1
; VF_2-NEXT:  Found an estimated cost of 2 for VF 2 For instruction:   store i8 %a2, ptr %p2, align 1
; VF_4-LABEL: Checking a loop in 'i8_factor_3'
; VF_4:       Found an estimated cost of 2 for VF 4 For instruction:   %l0 = load i8, ptr %p0, align 1
; VF_4-NEXT:  Found an estimated cost of 0 for VF 4 For instruction:   %l1 = load i8, ptr %p1, align 1
; VF_4-NEXT:  Found an estimated cost of 0 for VF 4 For instruction:   %l2 = load i8, ptr %p2, align 1
; VF_4:       Found an estimated cost of 0 for VF 4 For instruction:   store i8 %a0, ptr %p0, align 1
; VF_4:       Found an estimated cost of 0 for VF 4 For instruction:   store i8 %a1, ptr %p1, align 1
; VF_4-NEXT:  Found an estimated cost of 2 for VF 4 For instruction:   store i8 %a2, ptr %p2, align 1
; VF_8-LABEL: Checking a loop in 'i8_factor_3'
; VF_8:       Found an estimated cost of 2 for VF 8 For instruction:   %l0 = load i8, ptr %p0, align 1
; VF_8-NEXT:  Found an estimated cost of 0 for VF 8 For instruction:   %l1 = load i8, ptr %p1, align 1
; VF_8-NEXT:  Found an estimated cost of 0 for VF 8 For instruction:   %l2 = load i8, ptr %p2, align 1
; VF_8:       Found an estimated cost of 0 for VF 8 For instruction:   store i8 %a0, ptr %p0, align 1
; VF_8:       Found an estimated cost of 0 for VF 8 For instruction:   store i8 %a1, ptr %p1, align 1
; VF_8-NEXT:  Found an estimated cost of 2 for VF 8 For instruction:   store i8 %a2, ptr %p2, align 1
; VF_16-LABEL: Checking a loop in 'i8_factor_3'
; VF_16:       Found an estimated cost of 48 for VF 16 For instruction:   %l0 = load i8, ptr %p0, align 1
; VF_16-NEXT:  Found an estimated cost of 0 for VF 16 For instruction:   %l1 = load i8, ptr %p1, align 1
; VF_16-NEXT:  Found an estimated cost of 0 for VF 16 For instruction:   %l2 = load i8, ptr %p2, align 1
; VF_16:       Found an estimated cost of 0 for VF 16 For instruction:   store i8 %a0, ptr %p0, align 1
; VF_16:       Found an estimated cost of 0 for VF 16 For instruction:   store i8 %a1, ptr %p1, align 1
; VF_16-NEXT:  Found an estimated cost of 48 for VF 16 For instruction:   store i8 %a2, ptr %p2, align 1
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
