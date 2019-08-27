; RUN: opt -loop-vectorize -force-vector-width=2 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_2
; RUN: opt -loop-vectorize -force-vector-width=4 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_4
; RUN: opt -loop-vectorize -force-vector-width=8 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_8
; RUN: opt -loop-vectorize -force-vector-width=16 -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=VF_16
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1-m.main-none-eabi"

%i8.2 = type {i8, i8}
define void @i8_factor_2(%i8.2* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_8-LABEL:  Checking a loop in "i8_factor_2"
; VF_8:          Found an estimated cost of 2 for VF 8 For instruction: %tmp2 = load i8, i8* %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load i8, i8* %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i8 0, i8* %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 2 for VF 8 For instruction: store i8 0, i8* %tmp1, align 1
; VF_16-LABEL: Checking a loop in "i8_factor_2"
; VF_16:         Found an estimated cost of 2 for VF 16 For instruction: %tmp2 = load i8, i8* %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load i8, i8* %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i8 0, i8* %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 2 for VF 16 For instruction: store i8 0, i8* %tmp1, align 1
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i8.2, %i8.2* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i8.2, %i8.2* %data, i64 %i, i32 1
  %tmp2 = load i8, i8* %tmp0, align 1
  %tmp3 = load i8, i8* %tmp1, align 1
  store i8 0, i8* %tmp0, align 1
  store i8 0, i8* %tmp1, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i16.2 = type {i16, i16}
define void @i16_factor_2(%i16.2* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_4-LABEL:  Checking a loop in "i16_factor_2"
; VF_4:          Found an estimated cost of 2 for VF 4 For instruction: %tmp2 = load i16, i16* %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp3 = load i16, i16* %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i16 0, i16* %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 2 for VF 4 For instruction: store i16 0, i16* %tmp1, align 2
; VF_8-LABEL:  Checking a loop in "i16_factor_2"
; VF_8:          Found an estimated cost of 2 for VF 8 For instruction: %tmp2 = load i16, i16* %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load i16, i16* %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i16 0, i16* %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 2 for VF 8 For instruction: store i16 0, i16* %tmp1, align 2
; VF_16-LABEL: Checking a loop in "i16_factor_2"
; VF_16:         Found an estimated cost of 4 for VF 16 For instruction: %tmp2 = load i16, i16* %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load i16, i16* %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i16 0, i16* %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 4 for VF 16 For instruction: store i16 0, i16* %tmp1, align 2
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i16.2, %i16.2* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i16.2, %i16.2* %data, i64 %i, i32 1
  %tmp2 = load i16, i16* %tmp0, align 2
  %tmp3 = load i16, i16* %tmp1, align 2
  store i16 0, i16* %tmp0, align 2
  store i16 0, i16* %tmp1, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i32.2 = type {i32, i32}
define void @i32_factor_2(%i32.2* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in "i32_factor_2"
; VF_2:          Found an estimated cost of 2 for VF 2 For instruction: %tmp2 = load i32, i32* %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp3 = load i32, i32* %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store i32 0, i32* %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 2 for VF 2 For instruction: store i32 0, i32* %tmp1, align 4
; VF_4-LABEL:  Checking a loop in "i32_factor_2"
; VF_4:          Found an estimated cost of 2 for VF 4 For instruction: %tmp2 = load i32, i32* %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp3 = load i32, i32* %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i32 0, i32* %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 2 for VF 4 For instruction: store i32 0, i32* %tmp1, align 4
; VF_8-LABEL:  Checking a loop in "i32_factor_2"
; VF_8:          Found an estimated cost of 4 for VF 8 For instruction: %tmp2 = load i32, i32* %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp3 = load i32, i32* %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i32 0, i32* %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 4 for VF 8 For instruction: store i32 0, i32* %tmp1, align 4
; VF_16-LABEL: Checking a loop in "i32_factor_2"
; VF_16:         Found an estimated cost of 8 for VF 16 For instruction: %tmp2 = load i32, i32* %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp3 = load i32, i32* %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i32 0, i32* %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 8 for VF 16 For instruction: store i32 0, i32* %tmp1, align 4
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i32.2, %i32.2* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i32.2, %i32.2* %data, i64 %i, i32 1
  %tmp2 = load i32, i32* %tmp0, align 4
  %tmp3 = load i32, i32* %tmp1, align 4
  store i32 0, i32* %tmp0, align 4
  store i32 0, i32* %tmp1, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i8.3 = type {i8, i8, i8}
define void @i8_factor_3(%i8.3* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_8-LABEL:  Checking a loop in "i8_factor_3"
; VF_8:          Found an estimated cost of 408 for VF 8 For instruction: %tmp3 = load i8, i8* %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp4 = load i8, i8* %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load i8, i8* %tmp2, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i8 0, i8* %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i8 0, i8* %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 216 for VF 8 For instruction: store i8 0, i8* %tmp2, align 1
; VF_16-LABEL: Checking a loop in "i8_factor_3"
; VF_16:         Found an estimated cost of 1584 for VF 16 For instruction: %tmp3 = load i8, i8* %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp4 = load i8, i8* %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load i8, i8* %tmp2, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i8 0, i8* %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i8 0, i8* %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 816 for VF 16 For instruction: store i8 0, i8* %tmp2, align 1
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i8.3, %i8.3* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i8.3, %i8.3* %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i8.3, %i8.3* %data, i64 %i, i32 2
  %tmp3 = load i8, i8* %tmp0, align 1
  %tmp4 = load i8, i8* %tmp1, align 1
  %tmp5 = load i8, i8* %tmp2, align 1
  store i8 0, i8* %tmp0, align 1
  store i8 0, i8* %tmp1, align 1
  store i8 0, i8* %tmp2, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i16.3 = type {i16, i16, i16}
define void @i16_factor_3(%i16.3* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_4-LABEL:  Checking a loop in "i16_factor_3"
; VF_4:          Found an estimated cost of 108 for VF 4 For instruction: %tmp3 = load i16, i16* %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp4 = load i16, i16* %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp5 = load i16, i16* %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i16 0, i16* %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i16 0, i16* %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 60 for VF 4 For instruction: store i16 0, i16* %tmp2, align 2
; VF_8-LABEL:  Checking a loop in "i16_factor_3"
; VF_8:          Found an estimated cost of 408 for VF 8 For instruction: %tmp3 = load i16, i16* %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp4 = load i16, i16* %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load i16, i16* %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i16 0, i16* %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i16 0, i16* %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 216 for VF 8 For instruction: store i16 0, i16* %tmp2, align 2
; VF_16-LABEL: Checking a loop in "i16_factor_3"
; VF_16:         Found an estimated cost of 1584 for VF 16 For instruction: %tmp3 = load i16, i16* %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp4 = load i16, i16* %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load i16, i16* %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i16 0, i16* %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i16 0, i16* %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 816 for VF 16 For instruction: store i16 0, i16* %tmp2, align 2
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i16.3, %i16.3* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i16.3, %i16.3* %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i16.3, %i16.3* %data, i64 %i, i32 2
  %tmp3 = load i16, i16* %tmp0, align 2
  %tmp4 = load i16, i16* %tmp1, align 2
  %tmp5 = load i16, i16* %tmp2, align 2
  store i16 0, i16* %tmp0, align 2
  store i16 0, i16* %tmp1, align 2
  store i16 0, i16* %tmp2, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i32.3 = type {i32, i32, i32}
define void @i32_factor_3(%i32.3* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in "i32_factor_3"
; VF_2:          Found an estimated cost of 30 for VF 2 For instruction: %tmp3 = load i32, i32* %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp4 = load i32, i32* %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp5 = load i32, i32* %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store i32 0, i32* %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store i32 0, i32* %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 18 for VF 2 For instruction: store i32 0, i32* %tmp2, align 4
; VF_4-LABEL:  Checking a loop in "i32_factor_3"
; VF_4:          Found an estimated cost of 108 for VF 4 For instruction: %tmp3 = load i32, i32* %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp4 = load i32, i32* %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp5 = load i32, i32* %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i32 0, i32* %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i32 0, i32* %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 60 for VF 4 For instruction: store i32 0, i32* %tmp2, align 4
; VF_8-LABEL:  Checking a loop in "i32_factor_3"
; VF_8:          Found an estimated cost of 408 for VF 8 For instruction: %tmp3 = load i32, i32* %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp4 = load i32, i32* %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load i32, i32* %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i32 0, i32* %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i32 0, i32* %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 216 for VF 8 For instruction: store i32 0, i32* %tmp2, align 4
; VF_16-LABEL: Checking a loop in "i32_factor_3"
; VF_16:         Found an estimated cost of 1584 for VF 16 For instruction: %tmp3 = load i32, i32* %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp4 = load i32, i32* %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load i32, i32* %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i32 0, i32* %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i32 0, i32* %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 816 for VF 16 For instruction: store i32 0, i32* %tmp2, align 4
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i32.3, %i32.3* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i32.3, %i32.3* %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i32.3, %i32.3* %data, i64 %i, i32 2
  %tmp3 = load i32, i32* %tmp0, align 4
  %tmp4 = load i32, i32* %tmp1, align 4
  %tmp5 = load i32, i32* %tmp2, align 4
  store i32 0, i32* %tmp0, align 4
  store i32 0, i32* %tmp1, align 4
  store i32 0, i32* %tmp2, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}


%i8.4 = type {i8, i8, i8, i8}
define void @i8_factor_4(%i8.4* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_8-LABEL:  Checking a loop in "i8_factor_4"
; VF_8:          Found an estimated cost of 544 for VF 8 For instruction: %tmp4 = load i8, i8* %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load i8, i8* %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp6 = load i8, i8* %tmp2, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp7 = load i8, i8* %tmp3, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i8 0, i8* %tmp0, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i8 0, i8* %tmp1, align 1
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i8 0, i8* %tmp2, align 1
; VF_8-NEXT:     Found an estimated cost of 288 for VF 8 For instruction: store i8 0, i8* %tmp3, align 1
; VF_16-LABEL: Checking a loop in "i8_factor_4"
; VF_16:         Found an estimated cost of 2112 for VF 16 For instruction: %tmp4 = load i8, i8* %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load i8, i8* %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp6 = load i8, i8* %tmp2, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp7 = load i8, i8* %tmp3, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i8 0, i8* %tmp0, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i8 0, i8* %tmp1, align 1
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i8 0, i8* %tmp2, align 1
; VF_16-NEXT:    Found an estimated cost of 1088 for VF 16 For instruction: store i8 0, i8* %tmp3, align 1
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i8.4, %i8.4* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i8.4, %i8.4* %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i8.4, %i8.4* %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %i8.4, %i8.4* %data, i64 %i, i32 3
  %tmp4 = load i8, i8* %tmp0, align 1
  %tmp5 = load i8, i8* %tmp1, align 1
  %tmp6 = load i8, i8* %tmp2, align 1
  %tmp7 = load i8, i8* %tmp3, align 1
  store i8 0, i8* %tmp0, align 1
  store i8 0, i8* %tmp1, align 1
  store i8 0, i8* %tmp2, align 1
  store i8 0, i8* %tmp3, align 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i16.4 = type {i16, i16, i16, i16}
define void @i16_factor_4(%i16.4* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_4-LABEL:  Checking a loop in "i16_factor_4"
; VF_4:          Found an estimated cost of 144 for VF 4 For instruction: %tmp4 = load i16, i16* %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp5 = load i16, i16* %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp6 = load i16, i16* %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp7 = load i16, i16* %tmp3, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i16 0, i16* %tmp0, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i16 0, i16* %tmp1, align 2
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i16 0, i16* %tmp2, align 2
; VF_4-NEXT:     Found an estimated cost of 80 for VF 4 For instruction: store i16 0, i16* %tmp3, align 2
; VF_8-LABEL:  Checking a loop in "i16_factor_4"
; VF_8:          Found an estimated cost of 544 for VF 8 For instruction: %tmp4 = load i16, i16* %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load i16, i16* %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp6 = load i16, i16* %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp7 = load i16, i16* %tmp3, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i16 0, i16* %tmp0, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i16 0, i16* %tmp1, align 2
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i16 0, i16* %tmp2, align 2
; VF_8-NEXT:     Found an estimated cost of 288 for VF 8 For instruction: store i16 0, i16* %tmp3, align 2
; VF_16-LABEL: Checking a loop in "i16_factor_4"
; VF_16:         Found an estimated cost of 2112 for VF 16 For instruction: %tmp4 = load i16, i16* %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load i16, i16* %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp6 = load i16, i16* %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp7 = load i16, i16* %tmp3, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i16 0, i16* %tmp0, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i16 0, i16* %tmp1, align 2
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i16 0, i16* %tmp2, align 2
; VF_16-NEXT:    Found an estimated cost of 1088 for VF 16 For instruction: store i16 0, i16* %tmp3, align 2
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i16.4, %i16.4* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i16.4, %i16.4* %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i16.4, %i16.4* %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %i16.4, %i16.4* %data, i64 %i, i32 3
  %tmp4 = load i16, i16* %tmp0, align 2
  %tmp5 = load i16, i16* %tmp1, align 2
  %tmp6 = load i16, i16* %tmp2, align 2
  %tmp7 = load i16, i16* %tmp3, align 2
  store i16 0, i16* %tmp0, align 2
  store i16 0, i16* %tmp1, align 2
  store i16 0, i16* %tmp2, align 2
  store i16 0, i16* %tmp3, align 2
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

%i32.4 = type {i32, i32, i32, i32}
define void @i32_factor_4(%i32.4* %data, i64 %n) #0 {
entry:
  br label %for.body

; VF_2-LABEL:  Checking a loop in "i32_factor_4"
; VF_2:          Found an estimated cost of 40 for VF 2 For instruction: %tmp4 = load i32, i32* %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp5 = load i32, i32* %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp6 = load i32, i32* %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: %tmp7 = load i32, i32* %tmp3, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store i32 0, i32* %tmp0, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store i32 0, i32* %tmp1, align 4
; VF_2-NEXT:     Found an estimated cost of 0 for VF 2 For instruction: store i32 0, i32* %tmp2, align 4
; VF_2-NEXT:     Found an estimated cost of 24 for VF 2 For instruction: store i32 0, i32* %tmp3, align 4
; VF_4-LABEL:  Checking a loop in "i32_factor_4"
; VF_4:          Found an estimated cost of 144 for VF 4 For instruction: %tmp4 = load i32, i32* %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp5 = load i32, i32* %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp6 = load i32, i32* %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: %tmp7 = load i32, i32* %tmp3, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i32 0, i32* %tmp0, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i32 0, i32* %tmp1, align 4
; VF_4-NEXT:     Found an estimated cost of 0 for VF 4 For instruction: store i32 0, i32* %tmp2, align 4
; VF_4-NEXT:     Found an estimated cost of 80 for VF 4 For instruction: store i32 0, i32* %tmp3, align 4
; VF_8-LABEL:  Checking a loop in "i32_factor_4"
; VF_8:          Found an estimated cost of 544 for VF 8 For instruction: %tmp4 = load i32, i32* %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp5 = load i32, i32* %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp6 = load i32, i32* %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: %tmp7 = load i32, i32* %tmp3, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i32 0, i32* %tmp0, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i32 0, i32* %tmp1, align 4
; VF_8-NEXT:     Found an estimated cost of 0 for VF 8 For instruction: store i32 0, i32* %tmp2, align 4
; VF_8-NEXT:     Found an estimated cost of 288 for VF 8 For instruction: store i32 0, i32* %tmp3, align 4
; VF_16-LABEL: Checking a loop in "i32_factor_4"
; VF_16:         Found an estimated cost of 2112 for VF 16 For instruction: %tmp4 = load i32, i32* %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp5 = load i32, i32* %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp6 = load i32, i32* %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: %tmp7 = load i32, i32* %tmp3, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i32 0, i32* %tmp0, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i32 0, i32* %tmp1, align 4
; VF_16-NEXT:    Found an estimated cost of 0 for VF 16 For instruction: store i32 0, i32* %tmp2, align 4
; VF_16-NEXT:    Found an estimated cost of 1088 for VF 16 For instruction: store i32 0, i32* %tmp3, align 4
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds %i32.4, %i32.4* %data, i64 %i, i32 0
  %tmp1 = getelementptr inbounds %i32.4, %i32.4* %data, i64 %i, i32 1
  %tmp2 = getelementptr inbounds %i32.4, %i32.4* %data, i64 %i, i32 2
  %tmp3 = getelementptr inbounds %i32.4, %i32.4* %data, i64 %i, i32 3
  %tmp4 = load i32, i32* %tmp0, align 4
  %tmp5 = load i32, i32* %tmp1, align 4
  %tmp6 = load i32, i32* %tmp2, align 4
  %tmp7 = load i32, i32* %tmp3, align 4
  store i32 0, i32* %tmp0, align 4
  store i32 0, i32* %tmp1, align 4
  store i32 0, i32* %tmp2, align 4
  store i32 0, i32* %tmp3, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

attributes #0 = { "target-features"="+mve.fp" }
