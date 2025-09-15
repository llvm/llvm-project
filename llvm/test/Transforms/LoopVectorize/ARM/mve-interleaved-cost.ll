; RUN: opt -passes=loop-vectorize -force-vector-width=2 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_2
; RUN: opt -passes=loop-vectorize -force-vector-width=4 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_4
; RUN: opt -passes=loop-vectorize -force-vector-width=8 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_8
; RUN: opt -passes=loop-vectorize -force-vector-width=16 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_16
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-none-eabi"

; Factor 2

%i8.2 = type {i8, i8}
define void @i8_factor_2(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i8_factor_2'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp2 = load i8, ptr %tmp0, align 1
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp3 = load i8, ptr %tmp1, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp2, ptr %tmp0, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp3, ptr %tmp1, align 1
; VF_4-LABEL:  Checking a loop in 'i8_factor_2'
; VF_4:          Found an estimated cost of 4 for VF 4 For instruction: %tmp2 = load i8, ptr %tmp0, align 1
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp3 = load i8, ptr %tmp1, align 1
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i8 %tmp2, ptr %tmp0, align 1
; VF_4-NEXT:     Found an estimated cost of 4 for VF 4 For instruction: store i8 %tmp3, ptr %tmp1, align 1
; VF_8-LABEL:  Checking a loop in 'i8_factor_2'
; VF_8:          Found an estimated cost of 4 for VF 8 For instruction: %tmp2 = load i8, ptr %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load i8, ptr %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i8 %tmp2, ptr %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 4 for VF 8 For instruction: store i8 %tmp3, ptr %tmp1, align 1
; VF_16-LABEL: Checking a loop in 'i8_factor_2'
; VF_16:         Found an estimated cost of 4 for VF 16 For instruction: %tmp2 = load i8, ptr %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load i8, ptr %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i8 %tmp2, ptr %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 4 for VF 16 For instruction: store i8 %tmp3, ptr %tmp1, align 1
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
define void @i16_factor_2(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i16_factor_2'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp2 = load i16, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp3 = load i16, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp2, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp3, ptr %tmp1, align 2
; VF_4-LABEL:  Checking a loop in 'i16_factor_2'
; VF_4:          Found an estimated cost of 4 for VF 4 For instruction: %tmp2 = load i16, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp3 = load i16, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i16 %tmp2, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 4 for VF 4 For instruction: store i16 %tmp3, ptr %tmp1, align 2
; VF_8-LABEL:  Checking a loop in 'i16_factor_2'
; VF_8:          Found an estimated cost of 4 for VF 8 For instruction: %tmp2 = load i16, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load i16, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i16 %tmp2, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 4 for VF 8 For instruction: store i16 %tmp3, ptr %tmp1, align 2
; VF_16-LABEL: Checking a loop in 'i16_factor_2'
; VF_16:         Found an estimated cost of 8 for VF 16 For instruction: %tmp2 = load i16, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load i16, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i16 %tmp2, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 8 for VF 16 For instruction: store i16 %tmp3, ptr %tmp1, align 2
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
define void @i32_factor_2(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i32_factor_2'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp2 = load i32, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 12  for VF 2 For instruction: %tmp3 = load i32, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp2, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp3, ptr %tmp1, align 4
; VF_4-LABEL:  Checking a loop in 'i32_factor_2'
; VF_4:          Found an estimated cost of 4 for VF 4 For instruction: %tmp2 = load i32, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp3 = load i32, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i32 %tmp2, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 4 for VF 4 For instruction: store i32 %tmp3, ptr %tmp1, align 4
; VF_8-LABEL:  Checking a loop in 'i32_factor_2'
; VF_8:          Found an estimated cost of 8 for VF 8 For instruction: %tmp2 = load i32, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load i32, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i32 %tmp2, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 8 for VF 8 For instruction: store i32 %tmp3, ptr %tmp1, align 4
; VF_16-LABEL: Checking a loop in 'i32_factor_2'
; VF_16:         Found an estimated cost of 16 for VF 16 For instruction: %tmp2 = load i32, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load i32, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i32 %tmp2, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 16 for VF 16 For instruction: store i32 %tmp3, ptr %tmp1, align 4
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
define void @i64_factor_2(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i64_factor_2'
; VF_2:          Found an estimated cost of 22 for VF 2 For instruction: %tmp2 = load i64, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 22 for VF 2 For instruction: %tmp3 = load i64, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp2, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp3, ptr %tmp1, align 8
; VF_4-LABEL:  Checking a loop in 'i64_factor_2'
; VF_4:          Found an estimated cost of 44 for VF 4 For instruction: %tmp2 = load i64, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 44 for VF 4 For instruction: %tmp3 = load i64, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp2, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp3, ptr %tmp1, align 8
; VF_8-LABEL:  Checking a loop in 'i64_factor_2'
; VF_8:          Found an estimated cost of 88 for VF 8 For instruction: %tmp2 = load i64, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 88 for VF 8 For instruction: %tmp3 = load i64, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp2, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp3, ptr %tmp1, align 8
; VF_16-LABEL: Checking a loop in 'i64_factor_2'
; VF_16:         Found an estimated cost of 176 for VF 16 For instruction: %tmp2 = load i64, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 176 for VF 16 For instruction: %tmp3 = load i64, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp2, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp3, ptr %tmp1, align 8
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

%f16.2 = type {half, half}
define void @f16_factor_2(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f16_factor_2'
; VF_2:          Found an estimated cost of 6 for VF 2 For instruction: %tmp2 = load half, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp3 = load half, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store half %tmp2, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store half %tmp3, ptr %tmp1, align 2
; VF_4-LABEL:  Checking a loop in 'f16_factor_2'
; VF_4:          Found an estimated cost of 18 for VF 4 For instruction: %tmp2 = load half, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp3 = load half, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store half %tmp2, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 18 for VF 4 For instruction: store half %tmp3, ptr %tmp1, align 2
; VF_8-LABEL:  Checking a loop in 'f16_factor_2'
; VF_8:          Found an estimated cost of 4 for VF 8 For instruction: %tmp2 = load half, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load half, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store half %tmp2, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 4 for VF 8 For instruction: store half %tmp3, ptr %tmp1, align 2
; VF_16-LABEL: Checking a loop in 'f16_factor_2'
; VF_16:         Found an estimated cost of 8 for VF 16 For instruction: %tmp2 = load half, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load half, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store half %tmp2, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 8 for VF 16 For instruction: store half %tmp3, ptr %tmp1, align 2
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f16.2, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f16.2, ptr %data, i64 %i, i32 1
  %tmp2 = load half, ptr %tmp0, align 2
  %tmp3 = load half, ptr %tmp1, align 2
  store half %tmp2, ptr %tmp0, align 2
  store half %tmp3, ptr %tmp1, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%f32.2 = type {float, float}
define void @f32_factor_2(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f32_factor_2'
; VF_2:          Found an estimated cost of 10 for VF 2 For instruction: %tmp2 = load float, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp3 = load float, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store float %tmp2, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 10 for VF 2 For instruction: store float %tmp3, ptr %tmp1, align 4
; VF_4-LABEL:  Checking a loop in 'f32_factor_2'
; VF_4:          Found an estimated cost of 4 for VF 4 For instruction: %tmp2 = load float, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp3 = load float, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store float %tmp2, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 4 for VF 4 For instruction: store float %tmp3, ptr %tmp1, align 4
; VF_8-LABEL:  Checking a loop in 'f32_factor_2'
; VF_8:          Found an estimated cost of 8 for VF 8 For instruction: %tmp2 = load float, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load float, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store float %tmp2, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 8 for VF 8 For instruction: store float %tmp3, ptr %tmp1, align 4
; VF_16-LABEL: Checking a loop in 'f32_factor_2'
; VF_16:         Found an estimated cost of 16 for VF 16 For instruction: %tmp2 = load float, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load float, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store float %tmp2, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 16 for VF 16 For instruction: store float %tmp3, ptr %tmp1, align 4
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f32.2, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f32.2, ptr %data, i64 %i, i32 1
  %tmp2 = load float, ptr %tmp0, align 4
  %tmp3 = load float, ptr %tmp1, align 4
  store float %tmp2, ptr %tmp0, align 4
  store float %tmp3, ptr %tmp1, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%f64.2 = type {double, double}
define void @f64_factor_2(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f64_factor_2'
; VF_2:          Found an estimated cost of 6 for VF 2 For instruction: %tmp2 = load double, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp3 = load double, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp2, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp3, ptr %tmp1, align 8
; VF_4-LABEL:  Checking a loop in 'f64_factor_2'
; VF_4:          Found an estimated cost of 12 for VF 4 For instruction: %tmp2 = load double, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: %tmp3 = load double, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp2, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp3, ptr %tmp1, align 8
; VF_8-LABEL:  Checking a loop in 'f64_factor_2'
; VF_8:          Found an estimated cost of 24 for VF 8 For instruction: %tmp2 = load double, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: %tmp3 = load double, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp2, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp3, ptr %tmp1, align 8
; VF_16-LABEL: Checking a loop in 'f64_factor_2'
; VF_16:         Found an estimated cost of 48 for VF 16 For instruction: %tmp2 = load double, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: %tmp3 = load double, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp2, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp3, ptr %tmp1, align 8
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f64.2, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f64.2, ptr %data, i64 %i, i32 1
  %tmp2 = load double, ptr %tmp0, align 8
  %tmp3 = load double, ptr %tmp1, align 8
  store double %tmp2, ptr %tmp0, align 8
  store double %tmp3, ptr %tmp1, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}



; Factor 3

%i8.3 = type {i8, i8, i8}
define void @i8_factor_3(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i8_factor_3'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp3 = load i8, ptr %tmp0, align 1
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp4 = load i8, ptr %tmp1, align 1
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp5 = load i8, ptr %tmp2, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp3, ptr %tmp0, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp4, ptr %tmp1, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp5, ptr %tmp2, align 1
; VF_4-LABEL:  Checking a loop in 'i8_factor_3'
; VF_4:          Found an estimated cost of 24 for VF 4 For instruction: %tmp3 = load i8, ptr %tmp0, align 1
; VF_4-NEXT:     Found an estimated cost of 24 for VF 4 For instruction: %tmp4 = load i8, ptr %tmp1, align 1
; VF_4-NEXT:     Found an estimated cost of 24 for VF 4 For instruction: %tmp5 = load i8, ptr %tmp2, align 1
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i8 %tmp3,  ptr %tmp0, align 1
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i8 %tmp4, ptr %tmp1, align 1
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i8 %tmp5, ptr %tmp2, align 1
; VF_8-LABEL:  Checking a loop in 'i8_factor_3'
; VF_8:          Found an estimated cost of 48 for VF 8 For instruction: %tmp3 = load i8, ptr %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp4 = load i8, ptr %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp5 = load i8, ptr %tmp2, align 1
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i8 %tmp3, ptr %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i8 %tmp4, ptr %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i8 %tmp5, ptr %tmp2, align 1
; VF_16-LABEL: Checking a loop in 'i8_factor_3'
; VF_16:         Found an estimated cost of 96 for VF 16 For instruction: %tmp3 = load i8, ptr %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp4 = load i8, ptr %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp5 = load i8, ptr %tmp2, align 1
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i8 %tmp3, ptr %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i8 %tmp4, ptr %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i8 %tmp5, ptr %tmp2, align 1
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i8.3, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i8.3, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i8.3, ptr %data, i64 %i, i32 2
  %tmp3 = load i8, ptr %tmp0, align 1
  %tmp4 = load i8, ptr %tmp1, align 1
  %tmp5 = load i8, ptr %tmp2, align 1
  store i8 %tmp3, ptr %tmp0, align 1
  store i8 %tmp4, ptr %tmp1, align 1
  store i8 %tmp5, ptr %tmp2, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i16.3 = type {i16, i16, i16}
define void @i16_factor_3(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i16_factor_3'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp3 = load i16, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp4 = load i16, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp5 = load i16, ptr %tmp2, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp3, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp4, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp5, ptr %tmp2, align 2
; VF_4-LABEL:  Checking a loop in 'i16_factor_3'
; VF_4:          Found an estimated cost of 24 for VF 4 For instruction: %tmp3 = load i16, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 24 for VF 4 For instruction: %tmp4 = load i16, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 24 for VF 4 For instruction: %tmp5 = load i16, ptr %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i16 %tmp3, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i16 %tmp4, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i16 %tmp5, ptr %tmp2, align 2
; VF_8-LABEL:  Checking a loop in 'i16_factor_3'
; VF_8:          Found an estimated cost of 48 for VF 8 For instruction: %tmp3 = load i16, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp4 = load i16, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp5 = load i16, ptr %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i16 %tmp3, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i16 %tmp4, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i16 %tmp5, ptr %tmp2, align 2
; VF_16-LABEL: Checking a loop in 'i16_factor_3'
; VF_16:         Found an estimated cost of 96 for VF 16 For instruction: %tmp3 = load i16, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp4 = load i16, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp5 = load i16, ptr %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i16 %tmp3, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i16 %tmp4, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i16 %tmp5, ptr %tmp2, align 2
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i16.3, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i16.3, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i16.3, ptr %data, i64 %i, i32 2
  %tmp3 = load i16, ptr %tmp0, align 2
  %tmp4 = load i16, ptr %tmp1, align 2
  %tmp5 = load i16, ptr %tmp2, align 2
  store i16 %tmp3, ptr %tmp0, align 2
  store i16 %tmp4, ptr %tmp1, align 2
  store i16 %tmp5, ptr %tmp2, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i32.3 = type {i32, i32, i32}
define void @i32_factor_3(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i32_factor_3'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp3 = load i32, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp4 = load i32, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp5 = load i32, ptr %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp3, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp4, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp5, ptr %tmp2, align 4
; VF_4-LABEL:  Checking a loop in 'i32_factor_3'
; VF_4:          Found an estimated cost of 8 for VF 4 For instruction: %tmp3 = load i32, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp4 = load i32, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp5 = load i32, ptr %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i32 %tmp3, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i32 %tmp4, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i32 %tmp5, ptr %tmp2, align 4
; VF_8-LABEL:  Checking a loop in 'i32_factor_3'
; VF_8:          Found an estimated cost of 48 for VF 8 For instruction: %tmp3 = load i32, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp4 = load i32, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp5 = load i32, ptr %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i32 %tmp3, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i32 %tmp4, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i32 %tmp5, ptr %tmp2, align 4
; VF_16-LABEL: Checking a loop in 'i32_factor_3'
; VF_16:         Found an estimated cost of 96 for VF 16 For instruction: %tmp3 = load i32, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp4 = load i32, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp5 = load i32, ptr %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i32 %tmp3, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i32 %tmp4, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i32 %tmp5, ptr %tmp2, align 4
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i32.3, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i32.3, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i32.3, ptr %data, i64 %i, i32 2
  %tmp3 = load i32, ptr %tmp0, align 4
  %tmp4 = load i32, ptr %tmp1, align 4
  %tmp5 = load i32, ptr %tmp2, align 4
  store i32 %tmp3, ptr %tmp0, align 4
  store i32 %tmp4, ptr %tmp1, align 4
  store i32 %tmp5, ptr %tmp2, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i64.3 = type {i64, i64, i64}
define void @i64_factor_3(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i64_factor_3'
; VF_2:          Found an estimated cost of 22 for VF 2 For instruction: %tmp3 = load i64, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 22 for VF 2 For instruction: %tmp4 = load i64, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 22 for VF 2 For instruction: %tmp5 = load i64, ptr %tmp2, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp3, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp4, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp5, ptr %tmp2, align 8
; VF_4-LABEL:  Checking a loop in 'i64_factor_3'
; VF_4:          Found an estimated cost of 44 for VF 4 For instruction: %tmp3 = load i64, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 44 for VF 4 For instruction: %tmp4 = load i64, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 44 for VF 4 For instruction: %tmp5 = load i64, ptr %tmp2, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp3, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp4, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp5, ptr %tmp2, align 8
; VF_8-LABEL:  Checking a loop in 'i64_factor_3'
; VF_8:          Found an estimated cost of 88 for VF 8 For instruction: %tmp3 = load i64, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 88 for VF 8 For instruction: %tmp4 = load i64, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 88 for VF 8 For instruction: %tmp5 = load i64, ptr %tmp2, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp3, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp4, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp5, ptr %tmp2, align 8
; VF_16-LABEL: Checking a loop in 'i64_factor_3'
; VF_16:         Found an estimated cost of 176 for VF 16 For instruction: %tmp3 = load i64, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 176 for VF 16 For instruction: %tmp4 = load i64, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 176 for VF 16 For instruction: %tmp5 = load i64, ptr %tmp2, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp3, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp4, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp5, ptr %tmp2, align 8
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i64.3, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i64.3, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i64.3, ptr %data, i64 %i, i32 2
  %tmp3 = load i64, ptr %tmp0, align 8
  %tmp4 = load i64, ptr %tmp1, align 8
  %tmp5 = load i64, ptr %tmp2, align 8
  store i64 %tmp3, ptr %tmp0, align 8
  store i64 %tmp4, ptr %tmp1, align 8
  store i64 %tmp5, ptr %tmp2, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%f16.3 = type {half, half, half}
define void @f16_factor_3(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f16_factor_3'
; VF_2:          Found an estimated cost of 6 for VF 2 For instruction: %tmp3 = load half, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp4 = load half, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp5 = load half, ptr %tmp2, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store half %tmp3, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store half %tmp4, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store half %tmp5, ptr %tmp2, align 2
; VF_4-LABEL:  Checking a loop in 'f16_factor_3'
; VF_4:          Found an estimated cost of 28 for VF 4 For instruction: %tmp3 = load half, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp4 = load half, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp5 = load half, ptr %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store half %tmp3, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store half %tmp4, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 28 for VF 4 For instruction: store half %tmp5, ptr %tmp2, align 2
; VF_8-LABEL:  Checking a loop in 'f16_factor_3'
; VF_8:          Found an estimated cost of 56 for VF 8 For instruction: %tmp3 = load half, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp4 = load half, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load half, ptr %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store half %tmp3, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store half %tmp4, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 56 for VF 8 For instruction: store half %tmp5, ptr %tmp2, align 2
; VF_16-LABEL: Checking a loop in 'f16_factor_3'
; VF_16:         Found an estimated cost of 112 for VF 16 For instruction: %tmp3 = load half, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp4 = load half, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load half, ptr %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store half %tmp3, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store half %tmp4, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 112 for VF 16 For instruction: store half %tmp5, ptr %tmp2, align 2
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f16.3, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f16.3, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %f16.3, ptr %data, i64 %i, i32 2
  %tmp3 = load half, ptr %tmp0, align 2
  %tmp4 = load half, ptr %tmp1, align 2
  %tmp5 = load half, ptr %tmp2, align 2
  store half %tmp3, ptr %tmp0, align 2
  store half %tmp4, ptr %tmp1, align 2
  store half %tmp5, ptr %tmp2, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%f32.3 = type {float, float, float}
define void @f32_factor_3(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f32_factor_3'
; VF_2:          Found an estimated cost of 16 for VF 2 For instruction: %tmp3 = load float, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp4 = load float, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp5 = load float, ptr %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store float %tmp3, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store float %tmp4, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 16 for VF 2 For instruction: store float %tmp5, ptr %tmp2, align 4
; VF_4-LABEL:  Checking a loop in 'f32_factor_3'
; VF_4:          Found an estimated cost of 8 for VF 4 For instruction: %tmp3 = load float, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp4 = load float, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp5 = load float, ptr %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store float %tmp3, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store float %tmp4, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store float %tmp5, ptr %tmp2, align 4
; VF_8-LABEL:  Checking a loop in 'f32_factor_3'
; VF_8:          Found an estimated cost of 64 for VF 8 For instruction: %tmp3 = load float, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp4 = load float, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load float, ptr %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store float %tmp3, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store float %tmp4, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 64 for VF 8 For instruction: store float %tmp5, ptr %tmp2, align 4
; VF_16-LABEL: Checking a loop in 'f32_factor_3'
; VF_16:         Found an estimated cost of 128 for VF 16 For instruction: %tmp3 = load float, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp4 = load float, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load float, ptr %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store float %tmp3, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store float %tmp4, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 128 for VF 16 For instruction: store float %tmp5, ptr %tmp2, align 4
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f32.3, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f32.3, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %f32.3, ptr %data, i64 %i, i32 2
  %tmp3 = load float, ptr %tmp0, align 4
  %tmp4 = load float, ptr %tmp1, align 4
  %tmp5 = load float, ptr %tmp2, align 4
  store float %tmp3, ptr %tmp0, align 4
  store float %tmp4, ptr %tmp1, align 4
  store float %tmp5, ptr %tmp2, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%f64.3 = type {double, double, double}
define void @f64_factor_3(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f64_factor_3'
; VF_2:          Found an estimated cost of 6 for VF 2 For instruction: %tmp3 = load double, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp4 = load double, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp5 = load double, ptr %tmp2, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp3, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp4, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp5, ptr %tmp2, align 8
; VF_4-LABEL:  Checking a loop in 'f64_factor_3'
; VF_4:          Found an estimated cost of 12 for VF 4 For instruction: %tmp3 = load double, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: %tmp4 = load double, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: %tmp5 = load double, ptr %tmp2, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp3, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp4, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp5, ptr %tmp2, align 8
; VF_8-LABEL:  Checking a loop in 'f64_factor_3'
; VF_8:          Found an estimated cost of 24 for VF 8 For instruction: %tmp3 = load double, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: %tmp4 = load double, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: %tmp5 = load double, ptr %tmp2, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp3, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp4, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp5, ptr %tmp2, align 8
; VF_16-LABEL: Checking a loop in 'f64_factor_3'
; VF_16:         Found an estimated cost of 48 for VF 16 For instruction: %tmp3 = load double, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: %tmp4 = load double, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: %tmp5 = load double, ptr %tmp2, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp3, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp4, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp5, ptr %tmp2, align 8
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f64.3, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f64.3, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %f64.3, ptr %data, i64 %i, i32 2
  %tmp3 = load double, ptr %tmp0, align 8
  %tmp4 = load double, ptr %tmp1, align 8
  %tmp5 = load double, ptr %tmp2, align 8
  store double %tmp3, ptr %tmp0, align 8
  store double %tmp4, ptr %tmp1, align 8
  store double %tmp5, ptr %tmp2, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}


; Factor 4

%i8.4 = type {i8, i8, i8, i8}
define void @i8_factor_4(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i8_factor_4'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp4 = load i8, ptr %tmp0, align 1
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp5 = load i8, ptr %tmp1, align 1
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp6 = load i8, ptr %tmp2, align 1
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp7 = load i8, ptr %tmp3, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp4, ptr %tmp0, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp5, ptr %tmp1, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp6, ptr %tmp2, align 1
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i8 %tmp7, ptr %tmp3, align 1
; VF_4-LABEL: Checking a loop in 'i8_factor_4'
; VF_4:         Found an estimated cost of 24 for VF 4 For instruction: %tmp4 = load i8, ptr %tmp0, align 1
; VF_4-NEXT:    Found an estimated cost of 24 for VF 4 For instruction: %tmp5 = load i8, ptr %tmp1, align 1
; VF_4-NEXT:    Found an estimated cost of 24 for VF 4 For instruction: %tmp6 = load i8, ptr %tmp2, align 1
; VF_4-NEXT:    Found an estimated cost of 24 for VF 4 For instruction: %tmp7 = load i8, ptr %tmp3, align 1
; VF_4-NEXT:    Found an estimated cost of 8 for VF 4 For instruction: store i8 %tmp4, ptr %tmp0, align 1
; VF_4-NEXT:    Found an estimated cost of 8 for VF 4 For instruction: store i8 %tmp5, ptr %tmp1, align 1
; VF_4-NEXT:    Found an estimated cost of 8 for VF 4 For instruction: store i8 %tmp6, ptr %tmp2, align 1
; VF_4-NEXT:    Found an estimated cost of 8 for VF 4 For instruction: store i8 %tmp7, ptr %tmp3, align 1
; VF_8-LABEL:  Checking a loop in 'i8_factor_4'
; VF_8:          Found an estimated cost of 48 for VF 8 For instruction: %tmp4 = load i8, ptr %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp5 = load i8, ptr %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp6 = load i8, ptr %tmp2, align 1
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp7 = load i8, ptr %tmp3, align 1
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i8 %tmp4, ptr %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i8 %tmp5, ptr %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i8 %tmp6, ptr %tmp2, align 1
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i8 %tmp7, ptr %tmp3, align 1
; VF_16-LABEL: Checking a loop in 'i8_factor_4'
; VF_16:         Found an estimated cost of 96 for VF 16 For instruction: %tmp4 = load i8, ptr %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp5 = load i8, ptr %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp6 = load i8, ptr %tmp2, align 1
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp7 = load i8, ptr %tmp3, align 1
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i8 %tmp4, ptr %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i8 %tmp5, ptr %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i8 %tmp6, ptr %tmp2, align 1
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i8 %tmp7, ptr %tmp3, align 1
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i8.4, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i8.4, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i8.4, ptr %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %i8.4, ptr %data, i64 %i, i32 3
  %tmp4 = load i8, ptr %tmp0, align 1
  %tmp5 = load i8, ptr %tmp1, align 1
  %tmp6 = load i8, ptr %tmp2, align 1
  %tmp7 = load i8, ptr %tmp3, align 1
  store i8 %tmp4, ptr %tmp0, align 1
  store i8 %tmp5, ptr %tmp1, align 1
  store i8 %tmp6, ptr %tmp2, align 1
  store i8 %tmp7, ptr %tmp3, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i16.4 = type {i16, i16, i16, i16}
define void @i16_factor_4(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i16_factor_4'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp4 = load i16, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp5 = load i16, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp6 = load i16, ptr %tmp2, align 2
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp7 = load i16, ptr %tmp3, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp4, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp5, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp6, ptr %tmp2, align 2
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i16 %tmp7, ptr %tmp3, align 2
; VF_4-LABEL:  Checking a loop in 'i16_factor_4'
; VF_4:          Found an estimated cost of 24 for VF 4 For instruction: %tmp4 = load i16, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 24 for VF 4 For instruction: %tmp5 = load i16, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 24 for VF 4 For instruction: %tmp6 = load i16, ptr %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 24 for VF 4 For instruction: %tmp7 = load i16, ptr %tmp3, align 2
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i16 %tmp4, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i16 %tmp5, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i16 %tmp6, ptr %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i16 %tmp7, ptr %tmp3, align 2
; VF_8-LABEL:  Checking a loop in 'i16_factor_4'
; VF_8:          Found an estimated cost of 48 for VF 8 For instruction: %tmp4 = load i16, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp5 = load i16, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp6 = load i16, ptr %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp7 = load i16, ptr %tmp3, align 2
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i16 %tmp4, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i16 %tmp5, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i16 %tmp6, ptr %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i16 %tmp7, ptr %tmp3, align 2
; VF_16-LABEL: Checking a loop in 'i16_factor_4'
; VF_16:         Found an estimated cost of 96 for VF 16 For instruction: %tmp4 = load i16, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp5 = load i16, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp6 = load i16, ptr %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp7 = load i16, ptr %tmp3, align 2
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i16 %tmp4, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i16 %tmp5, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i16 %tmp6, ptr %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i16 %tmp7, ptr %tmp3, align 2
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i16.4, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i16.4, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i16.4, ptr %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %i16.4, ptr %data, i64 %i, i32 3
  %tmp4 = load i16, ptr %tmp0, align 2
  %tmp5 = load i16, ptr %tmp1, align 2
  %tmp6 = load i16, ptr %tmp2, align 2
  %tmp7 = load i16, ptr %tmp3, align 2
  store i16 %tmp4, ptr %tmp0, align 2
  store i16 %tmp5, ptr %tmp1, align 2
  store i16 %tmp6, ptr %tmp2, align 2
  store i16 %tmp7, ptr %tmp3, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i32.4 = type {i32, i32, i32, i32}
define void @i32_factor_4(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i32_factor_4'
; VF_2:          Found an estimated cost of 12 for VF 2 For instruction: %tmp4 = load i32, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp5 = load i32, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp6 = load i32, ptr %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 12 for VF 2 For instruction: %tmp7 = load i32, ptr %tmp3, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp4, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp5, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp6, ptr %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store i32 %tmp7, ptr %tmp3, align 4
; VF_4-LABEL:  Checking a loop in 'i32_factor_4'
; VF_4:          Found an estimated cost of 8 for VF 4 For instruction: %tmp4 = load i32, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp5 = load i32, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp6 = load i32, ptr %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp7 = load i32, ptr %tmp3, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i32 %tmp4, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i32 %tmp5, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i32 %tmp6, ptr %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store i32 %tmp7, ptr %tmp3, align 4
; VF_8-LABEL:  Checking a loop in 'i32_factor_4'
; VF_8:          Found an estimated cost of 48 for VF 8 For instruction: %tmp4 = load i32, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp5 = load i32, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp6 = load i32, ptr %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 48 for VF 8 For instruction: %tmp7 = load i32, ptr %tmp3, align 4
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i32 %tmp4, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i32 %tmp5, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i32 %tmp6, ptr %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store i32 %tmp7, ptr %tmp3, align 4
; VF_16-LABEL: Checking a loop in 'i32_factor_4'
; VF_16:         Found an estimated cost of 96 for VF 16 For instruction: %tmp4 = load i32, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp5 = load i32, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp6 = load i32, ptr %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 96 for VF 16 For instruction: %tmp7 = load i32, ptr %tmp3, align 4
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i32 %tmp4, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i32 %tmp5, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i32 %tmp6, ptr %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store i32 %tmp7, ptr %tmp3, align 4
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i32.4, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i32.4, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i32.4, ptr %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %i32.4, ptr %data, i64 %i, i32 3
  %tmp4 = load i32, ptr %tmp0, align 4
  %tmp5 = load i32, ptr %tmp1, align 4
  %tmp6 = load i32, ptr %tmp2, align 4
  %tmp7 = load i32, ptr %tmp3, align 4
  store i32 %tmp4, ptr %tmp0, align 4
  store i32 %tmp5, ptr %tmp1, align 4
  store i32 %tmp6, ptr %tmp2, align 4
  store i32 %tmp7, ptr %tmp3, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i64.4 = type {i64, i64, i64, i64}
define void @i64_factor_4(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'i64_factor_4'
; VF_2:          Found an estimated cost of 22 for VF 2 For instruction: %tmp4 = load i64, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 22 for VF 2 For instruction: %tmp5 = load i64, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 22 for VF 2 For instruction: %tmp6 = load i64, ptr %tmp2, align 8
; VF_2-NEXT:     Found an estimated cost of 22 for VF 2 For instruction: %tmp7 = load i64, ptr %tmp3, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp4, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp5, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp6, ptr %tmp2, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: store i64 %tmp7, ptr %tmp3, align 8
; VF_4-LABEL:  Checking a loop in 'i64_factor_4'
; VF_4:          Found an estimated cost of 44 for VF 4 For instruction: %tmp4 = load i64, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 44 for VF 4 For instruction: %tmp5 = load i64, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 44 for VF 4 For instruction: %tmp6 = load i64, ptr %tmp2, align 8
; VF_4-NEXT:     Found an estimated cost of 44 for VF 4 For instruction: %tmp7 = load i64, ptr %tmp3, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp4, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp5, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp6, ptr %tmp2, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: store i64 %tmp7, ptr %tmp3, align 8
; VF_8-LABEL:  Checking a loop in 'i64_factor_4'
; VF_8:          Found an estimated cost of 88 for VF 8 For instruction: %tmp4 = load i64, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 88 for VF 8 For instruction: %tmp5 = load i64, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 88 for VF 8 For instruction: %tmp6 = load i64, ptr %tmp2, align 8
; VF_8-NEXT:     Found an estimated cost of 88 for VF 8 For instruction: %tmp7 = load i64, ptr %tmp3, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp4, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp5, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp6, ptr %tmp2, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: store i64 %tmp7, ptr %tmp3, align 8
; VF_16-LABEL: Checking a loop in 'i64_factor_4'
; VF_16:         Found an estimated cost of 176 for VF 16 For instruction: %tmp4 = load i64, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 176 for VF 16 For instruction: %tmp5 = load i64, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 176 for VF 16 For instruction: %tmp6 = load i64, ptr %tmp2, align 8
; VF_16-NEXT:    Found an estimated cost of 176 for VF 16 For instruction: %tmp7 = load i64, ptr %tmp3, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp4, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp5, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp6, ptr %tmp2, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: store i64 %tmp7, ptr %tmp3, align 8
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i64.4, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i64.4, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i64.4, ptr %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %i64.4, ptr %data, i64 %i, i32 3
  %tmp4 = load i64, ptr %tmp0, align 8
  %tmp5 = load i64, ptr %tmp1, align 8
  %tmp6 = load i64, ptr %tmp2, align 8
  %tmp7 = load i64, ptr %tmp3, align 8
  store i64 %tmp4, ptr %tmp0, align 8
  store i64 %tmp5, ptr %tmp1, align 8
  store i64 %tmp6, ptr %tmp2, align 8
  store i64 %tmp7, ptr %tmp3, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%f16.4 = type {half, half, half, half}
define void @f16_factor_4(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f16_factor_4'
; VF_2:          Found an estimated cost of 18 for VF 2 For instruction: %tmp4 = load half, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp5 = load half, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp6 = load half, ptr %tmp2, align 2
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp7 = load half, ptr %tmp3, align 2
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store half %tmp4, ptr %tmp0, align 2
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store half %tmp5, ptr %tmp1, align 2
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store half %tmp6, ptr %tmp2, align 2
; VF_2-NEXT:     Found an estimated cost of 18 for VF 2 For instruction: store half %tmp7, ptr %tmp3, align 2
; VF_4-LABEL:  Checking a loop in 'f16_factor_4'
; VF_4:          Found an estimated cost of 36 for VF 4 For instruction: %tmp4 = load half, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp5 = load half, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp6 = load half, ptr %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp7 = load half, ptr %tmp3, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store half %tmp4, ptr %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store half %tmp5, ptr %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store half %tmp6, ptr %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 36 for VF 4 For instruction: store half %tmp7, ptr %tmp3, align 2
; VF_8-LABEL:  Checking a loop in 'f16_factor_4'
; VF_8:          Found an estimated cost of 72 for VF 8 For instruction: %tmp4 = load half, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load half, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp6 = load half, ptr %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp7 = load half, ptr %tmp3, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store half %tmp4, ptr %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store half %tmp5, ptr %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store half %tmp6, ptr %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 72 for VF 8 For instruction: store half %tmp7, ptr %tmp3, align 2
; VF_16-LABEL: Checking a loop in 'f16_factor_4'
; VF_16:         Found an estimated cost of 144 for VF 16 For instruction: %tmp4 = load half, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load half, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp6 = load half, ptr %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp7 = load half, ptr %tmp3, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store half %tmp4, ptr %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store half %tmp5, ptr %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store half %tmp6, ptr %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 144 for VF 16 For instruction: store half %tmp7, ptr %tmp3, align 2
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f16.4, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f16.4, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %f16.4, ptr %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %f16.4, ptr %data, i64 %i, i32 3
  %tmp4 = load half, ptr %tmp0, align 2
  %tmp5 = load half, ptr %tmp1, align 2
  %tmp6 = load half, ptr %tmp2, align 2
  %tmp7 = load half, ptr %tmp3, align 2
  store half %tmp4, ptr %tmp0, align 2
  store half %tmp5, ptr %tmp1, align 2
  store half %tmp6, ptr %tmp2, align 2
  store half %tmp7, ptr %tmp3, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%f32.4 = type {float, float, float, float}
define void @f32_factor_4(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f32_factor_4'
; VF_2:          Found an estimated cost of 20 for VF 2 For instruction: %tmp4 = load float, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp5 = load float, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp6 = load float, ptr %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp7 = load float, ptr %tmp3, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store float %tmp4, ptr %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store float %tmp5, ptr %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store float %tmp6, ptr %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 20 for VF 2 For instruction: store float %tmp7, ptr %tmp3, align 4
; VF_4-LABEL:  Checking a loop in 'f32_factor_4'
; VF_4:          Found an estimated cost of 8 for VF 4 For instruction: %tmp4 = load float, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp5 = load float, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp6 = load float, ptr %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: %tmp7 = load float, ptr %tmp3, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store float %tmp4, ptr %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store float %tmp5, ptr %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store float %tmp6, ptr %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store float %tmp7, ptr %tmp3, align 4
; VF_8-LABEL:  Checking a loop in 'f32_factor_4'
; VF_8:          Found an estimated cost of 80 for VF 8 For instruction: %tmp4 = load float, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load float, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp6 = load float, ptr %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp7 = load float, ptr %tmp3, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store float %tmp4, ptr %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store float %tmp5, ptr %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store float %tmp6, ptr %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 80 for VF 8 For instruction: store float %tmp7, ptr %tmp3, align 4
; VF_16-LABEL: Checking a loop in 'f32_factor_4'
; VF_16:         Found an estimated cost of 160 for VF 16 For instruction: %tmp4 = load float, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load float, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp6 = load float, ptr %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp7 = load float, ptr %tmp3, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store float %tmp4, ptr %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store float %tmp5, ptr %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store float %tmp6, ptr %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 160 for VF 16 For instruction: store float %tmp7, ptr %tmp3, align 4
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f32.4, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f32.4, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %f32.4, ptr %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %f32.4, ptr %data, i64 %i, i32 3
  %tmp4 = load float, ptr %tmp0, align 4
  %tmp5 = load float, ptr %tmp1, align 4
  %tmp6 = load float, ptr %tmp2, align 4
  %tmp7 = load float, ptr %tmp3, align 4
  store float %tmp4, ptr %tmp0, align 4
  store float %tmp5, ptr %tmp1, align 4
  store float %tmp6, ptr %tmp2, align 4
  store float %tmp7, ptr %tmp3, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%f64.4 = type {double, double, double, double}
define void @f64_factor_4(ptr %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in 'f64_factor_4'
; VF_2:          Found an estimated cost of 6 for VF 2 For instruction: %tmp4 = load double, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp5 = load double, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp6 = load double, ptr %tmp2, align 8
; VF_2-NEXT:     Found an estimated cost of 6 for VF 2 For instruction: %tmp7 = load double, ptr %tmp3, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp4, ptr %tmp0, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp5, ptr %tmp1, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp6, ptr %tmp2, align 8
; VF_2-NEXT:     Found an estimated cost of 4 for VF 2 For instruction: store double %tmp7, ptr %tmp3, align 8
; VF_4-LABEL:  Checking a loop in 'f64_factor_4'
; VF_4:          Found an estimated cost of 12 for VF 4 For instruction: %tmp4 = load double, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: %tmp5 = load double, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: %tmp6 = load double, ptr %tmp2, align 8
; VF_4-NEXT:     Found an estimated cost of 12 for VF 4 For instruction: %tmp7 = load double, ptr %tmp3, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp4, ptr %tmp0, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp5, ptr %tmp1, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp6, ptr %tmp2, align 8
; VF_4-NEXT:     Found an estimated cost of 8 for VF 4 For instruction: store double %tmp7, ptr %tmp3, align 8
; VF_8-LABEL:  Checking a loop in 'f64_factor_4'
; VF_8:          Found an estimated cost of 24 for VF 8 For instruction: %tmp4 = load double, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: %tmp5 = load double, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: %tmp6 = load double, ptr %tmp2, align 8
; VF_8-NEXT:     Found an estimated cost of 24 for VF 8 For instruction: %tmp7 = load double, ptr %tmp3, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp4, ptr %tmp0, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp5, ptr %tmp1, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp6, ptr %tmp2, align 8
; VF_8-NEXT:     Found an estimated cost of 16 for VF 8 For instruction: store double %tmp7, ptr %tmp3, align 8
; VF_16-LABEL: Checking a loop in 'f64_factor_4'
; VF_16:         Found an estimated cost of 48 for VF 16 For instruction: %tmp4 = load double, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: %tmp5 = load double, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: %tmp6 = load double, ptr %tmp2, align 8
; VF_16-NEXT:    Found an estimated cost of 48 for VF 16 For instruction: %tmp7 = load double, ptr %tmp3, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp4, ptr %tmp0, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp5, ptr %tmp1, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp6, ptr %tmp2, align 8
; VF_16-NEXT:    Found an estimated cost of 32 for VF 16 For instruction: store double %tmp7, ptr %tmp3, align 8
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %f64.4, ptr %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %f64.4, ptr %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %f64.4, ptr %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %f64.4, ptr %data, i64 %i, i32 3
  %tmp4 = load double, ptr %tmp0, align 8
  %tmp5 = load double, ptr %tmp1, align 8
  %tmp6 = load double, ptr %tmp2, align 8
  %tmp7 = load double, ptr %tmp3, align 8
  store double %tmp4, ptr %tmp0, align 8
  store double %tmp5, ptr %tmp1, align 8
  store double %tmp6, ptr %tmp2, align 8
  store double %tmp7, ptr %tmp3, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

attributes #0 = { "target-features"="+mve.fp" }
