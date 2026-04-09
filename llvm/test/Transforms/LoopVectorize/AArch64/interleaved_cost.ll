; RUN: opt -passes=loop-vectorize -force-vector-width=2 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_2
; RUN: opt -passes=loop-vectorize -force-vector-width=4 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_4
; RUN: opt -passes=loop-vectorize -force-vector-width=8 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_8
; RUN: opt -passes=loop-vectorize -force-vector-width=16 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_16
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

%i8.2 = type {i8, i8}
define void @i8_factor_2(ptr %data, i64 %n) {
entry:
  br label %for.body

; VF_8-LABEL:  Checking a loop in 'i8_factor_2'
; VF_8:          Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_8:          Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
; VF_16-LABEL: Checking a loop in 'i8_factor_2'
; VF_16:         Cost of 2 for VF 16: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_16:         Cost of 2 for VF 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i8.2, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i8.2, ptr %data, i64 %i, i32 1
  %tmp2 = load i8, ptr %tmp0, align 1
  %tmp3 = load i8, ptr %tmp1, align 1
  store i8 %tmp2, ptr %tmp0, align 1
  store i8 %tmp3, ptr %tmp1, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i16.2 = type {i16, i16}
define void @i16_factor_2(ptr %data, i64 %n) {
entry:
  br label %for.body

; VF_4-LABEL: Checking a loop in 'i16_factor_2'
; VF_4:          Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_4:          Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
; VF_8-LABEL:  Checking a loop in 'i16_factor_2'
; VF_8:          Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_8:          Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
; VF_16-LABEL: Checking a loop in 'i16_factor_2'
; VF_16:         Cost of 4 for VF 16: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_16:         Cost of 4 for VF 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i16.2, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i16.2, ptr %data, i64 %i, i32 1
  %tmp2 = load i16, ptr %tmp0, align 2
  %tmp3 = load i16, ptr %tmp1, align 2
  store i16 %tmp2, ptr %tmp0, align 2
  store i16 %tmp3, ptr %tmp1, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i32.2 = type {i32, i32}
define void @i32_factor_2(ptr %data, i64 %n) {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i32_factor_2'
; VF_2:          Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_2:          Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
; VF_4-LABEL:  Checking a loop in 'i32_factor_2'
; VF_4:          Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_4:          Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
; VF_8-LABEL:  Checking a loop in 'i32_factor_2'
; VF_8:          Cost of 4 for VF 8: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_8:          Cost of 4 for VF 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
; VF_16-LABEL: Checking a loop in 'i32_factor_2'
; VF_16:         Cost of 8 for VF 16: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_16:         Cost of 8 for VF 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i32.2, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i32.2, ptr %data, i64 %i, i32 1
  %tmp2 = load i32, ptr %tmp0, align 4
  %tmp3 = load i32, ptr %tmp1, align 4
  store i32 %tmp2, ptr %tmp0, align 4
  store i32 %tmp3, ptr %tmp1, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i64.2 = type {i64, i64}
define void @i64_factor_2(ptr %data, i64 %n) {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i64_factor_2'
; VF_2:          Cost of 1 for VF 2: WIDEN ir<%tmp2> = load ir<%tmp0>
; VF_2-NEXT:     Cost of 1 for VF 2: WIDEN store ir<%tmp0>, ir<%tmp2>
; VF_4-LABEL:  Checking a loop in 'i64_factor_2'
; VF_4:          Cost of 4 for VF 4: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_4:          Cost of 4 for VF 4: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
; VF_8-LABEL:  Checking a loop in 'i64_factor_2'
; VF_8:          Cost of 8 for VF 8: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_8:          Cost of 8 for VF 8: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
; VF_16-LABEL: Checking a loop in 'i64_factor_2'
; VF_16:         Cost of 16 for VF 16: INTERLEAVE-GROUP with factor 2 at %tmp2, ir<%tmp0>
; VF_16:         Cost of 16 for VF 16: INTERLEAVE-GROUP with factor 2 at <badref>, ir<%tmp0>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i64.2, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i64.2, ptr %data, i64 %i, i32 1
  %tmp2 = load i64, ptr %tmp0, align 8
  %tmp3 = load i64, ptr %tmp1, align 8
  store i64 %tmp2, ptr %tmp0, align 8
  store i64 %tmp3, ptr %tmp1, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i64.8 = type {i64, i64, i64, i64, i64, i64, i64, i64}
define void @i64_factor_8(ptr %data, i64 %n) {
entry:
  br label %for.body

; The interleave factor in this test is 8, which is greater than the maximum
; allowed factor for AArch64 (4). Thus, we will fall back to the basic TTI
; implementation for determining the cost of the interleaved load group. The
; stores do not form a legal interleaved group because the group would contain
; gaps.
;
; VF_2-LABEL: Checking a loop in 'i64_factor_8'
; VF_2:         Cost of 8 for VF 2: REPLICATE ir<%tmp2> = load ir<%tmp0>
; VF_2-NEXT:    Cost of 8 for VF 2: REPLICATE ir<%tmp3> = load ir<%tmp1>
; VF_2-NEXT:    Cost of 8 for VF 2: REPLICATE store ir<%tmp2>, ir<%tmp0>
; VF_2-NEXT:    Cost of 8 for VF 2: REPLICATE store ir<%tmp3>, ir<%tmp1>
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i64.8, ptr %data, i64 %i, i32 2
  %tmp1 = getelementptr inbounds %i64.8, ptr %data, i64 %i, i32 6
  %tmp2 = load i64, ptr %tmp0, align 8
  %tmp3 = load i64, ptr %tmp1, align 8
  store i64 %tmp2, ptr %tmp0, align 8
  store i64 %tmp3, ptr %tmp1, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
