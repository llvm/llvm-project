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
; VF_8:          Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_8:          Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_16-LABEL: Checking a loop in 'i8_factor_2'
; VF_16:         Cost of 2 for VF 16: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_16:         Cost of 2 for VF 16: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
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
; VF_4:          Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_4:          Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_8-LABEL:  Checking a loop in 'i16_factor_2'
; VF_8:          Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_8:          Cost of 2 for VF 8: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_16-LABEL: Checking a loop in 'i16_factor_2'
; VF_16:         Cost of 4 for VF 16: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_16:         Cost of 4 for VF 16: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
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
; VF_2:          Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_2:          Cost of 2 for VF 2: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_4-LABEL:  Checking a loop in 'i32_factor_2'
; VF_4:          Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_4:          Cost of 2 for VF 4: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_8-LABEL:  Checking a loop in 'i32_factor_2'
; VF_8:          Cost of 4 for VF 8: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_8:          Cost of 4 for VF 8: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_16-LABEL: Checking a loop in 'i32_factor_2'
; VF_16:         Cost of 8 for VF 16: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_16:         Cost of 8 for VF 16: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
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
; VF_4:          Cost of 4 for VF 4: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_4:          Cost of 4 for VF 4: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_8-LABEL:  Checking a loop in 'i64_factor_2'
; VF_8:          Cost of 8 for VF 8: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_8:          Cost of 8 for VF 8: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_16-LABEL: Checking a loop in 'i64_factor_2'
; VF_16:         Cost of 16 for VF 16: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
; VF_16:         Cost of 16 for VF 16: INTERLEAVE-GROUP with factor 2, ir<%tmp0>
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

; VF_16-LABEL:  Checking a loop in 'i8_factor_6'
; VF_16:         Cost of 6 for VF 16: INTERLEAVE-GROUP with factor 6, ir<%arrayidx>
; VF_16:         Cost of 6 for VF 16: INTERLEAVE-GROUP with factor 6, ir<%arrayidx>
define void @i8_factor_6(ptr %p, ptr %o0, ptr %o1, ptr %o2, ptr %o3, ptr %o4, ptr %o5, i32 noundef %n) #0 {
entry:
  %cmp51 = icmp sgt i32 %n, 0
  br i1 %cmp51, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = mul nuw nsw i64 %indvars.iv, 6
  %arrayidx = getelementptr inbounds nuw i8, ptr %p, i64 %0
  %1 = load i8, ptr %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %o0, i64 %indvars.iv
  store i8 %1, ptr %arrayidx2, align 1
  %arrayidx6 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 1
  %2 = load i8, ptr %arrayidx6, align 1
  %arrayidx8 = getelementptr inbounds nuw i8, ptr %o1, i64 %indvars.iv
  store i8 %2, ptr %arrayidx8, align 1
  %arrayidx12 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 2
  %3 = load i8, ptr %arrayidx12, align 1
  %arrayidx14 = getelementptr inbounds nuw i8, ptr %o2, i64 %indvars.iv
  store i8 %3, ptr %arrayidx14, align 1
  %arrayidx18 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 3
  %4 = load i8, ptr %arrayidx18, align 1
  %arrayidx20 = getelementptr inbounds nuw i8, ptr %o3, i64 %indvars.iv
  store i8 %4, ptr %arrayidx20, align 1
  %arrayidx24 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 4
  %5 = load i8, ptr %arrayidx24, align 1
  %arrayidx26 = getelementptr inbounds nuw i8, ptr %o4, i64 %indvars.iv
  store i8 %5, ptr %arrayidx26, align 1
  %arrayidx30 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 5
  %6 = load i8, ptr %arrayidx30, align 1
  %arrayidx32 = getelementptr inbounds nuw i8, ptr %o5, i64 %indvars.iv
  store i8 %6, ptr %arrayidx32, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}

; VF_16-LABEL:  Checking a loop in 'i8_factor_8'
; VF_16:        Cost of 8 for VF 16: INTERLEAVE-GROUP with factor 8, ir<%arrayidx>
; VF_16:        Cost of 8 for VF 16: INTERLEAVE-GROUP with factor 8, ir<%arrayidx>
define void @i8_factor_8(ptr %p, ptr %o0, ptr %o1, ptr %o2, ptr %o3, ptr %o4, ptr %o5, ptr %o6, ptr %o7, i32 noundef %n) #0 {
entry:
  %cmp69 = icmp sgt i32 %n, 0
  br i1 %cmp69, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds nuw i8, ptr %p, i64 %0
  %1 = load i8, ptr %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %o0, i64 %indvars.iv
  store i8 %1, ptr %arrayidx2, align 1
  %arrayidx6 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 1
  %2 = load i8, ptr %arrayidx6, align 1
  %arrayidx8 = getelementptr inbounds nuw i8, ptr %o1, i64 %indvars.iv
  store i8 %2, ptr %arrayidx8, align 1
  %arrayidx12 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 2
  %3 = load i8, ptr %arrayidx12, align 1
  %arrayidx14 = getelementptr inbounds nuw i8, ptr %o2, i64 %indvars.iv
  store i8 %3, ptr %arrayidx14, align 1
  %arrayidx18 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 3
  %4 = load i8, ptr %arrayidx18, align 1
  %arrayidx20 = getelementptr inbounds nuw i8, ptr %o3, i64 %indvars.iv
  store i8 %4, ptr %arrayidx20, align 1
  %arrayidx24 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 4
  %5 = load i8, ptr %arrayidx24, align 1
  %arrayidx26 = getelementptr inbounds nuw i8, ptr %o4, i64 %indvars.iv
  store i8 %5, ptr %arrayidx26, align 1
  %arrayidx30 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 5
  %6 = load i8, ptr %arrayidx30, align 1
  %arrayidx32 = getelementptr inbounds nuw i8, ptr %o5, i64 %indvars.iv
  store i8 %6, ptr %arrayidx32, align 1
  %arrayidx36 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 6
  %7 = load i8, ptr %arrayidx36, align 1
  %arrayidx38 = getelementptr inbounds nuw i8, ptr %o6, i64 %indvars.iv
  store i8 %7, ptr %arrayidx38, align 1
  %arrayidx42 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 7
  %8 = load i8, ptr %arrayidx42, align 1
  %arrayidx44 = getelementptr inbounds nuw i8, ptr %o7, i64 %indvars.iv
  store i8 %8, ptr %arrayidx44, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
