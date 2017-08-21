; RUN: opt < %s -mcpu=corei7 -O1 -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=O1
; RUN: opt < %s -mcpu=corei7 -O2 -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=O2
; RUN: opt < %s -mcpu=corei7 -O3 -S -unroll-threshold=150 -unroll-allow-partial=0 | FileCheck %s --check-prefix=O3
; RUN: opt < %s -mcpu=corei7 -O3 -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=O3DEFAULT
; RUN: opt < %s -mcpu=corei7 -Os -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=Os
; RUN: opt < %s -mcpu=corei7 -Oz -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=Oz
; RUN: opt < %s -mcpu=corei7 -O1 -vectorize-loops -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=O1VEC
; RUN: opt < %s -mcpu=corei7 -Oz -vectorize-loops -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=OzVEC
; RUN: opt < %s -mcpu=corei7 -O1 -loop-vectorize -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=O1VEC2
; RUN: opt < %s -mcpu=corei7 -Oz -loop-vectorize -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=OzVEC2
; RUN: opt < %s -mcpu=corei7 -O3 -unroll-threshold=150 -disable-loop-vectorization -S -unroll-allow-partial=0 | FileCheck %s --check-prefix=O3DIS

; This file tests the llvm.loop.vectorize.enable metadata forcing
; vectorization even when optimization levels are too low, or when
; vectorization is disabled.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; O1-LABEL: @enabled(
; O1: store <4 x i32>
; O1: ret i32
; O2-LABEL: @enabled(
; O2: store <4 x i32>
; O2: ret i32
; O3-LABEL: @enabled(
; O3: store <4 x i32>
; O3: ret i32
; O3DEFAULT-LABEL: @enabled(
; O3DEFAULT: store <4 x i32>
; O3DEFAULT: ret i32
; Pragma always wins!
; O3DIS-LABEL: @enabled(
; O3DIS: store <4 x i32>
; O3DIS: ret i32
; Os-LABEL: @enabled(
; Os: store <4 x i32>
; Os: ret i32
; Oz-LABEL: @enabled(
; Oz: store <4 x i32>
; Oz: ret i32
; O1VEC-LABEL: @enabled(
; O1VEC: store <4 x i32>
; O1VEC: ret i32
; OzVEC-LABEL: @enabled(
; OzVEC: store <4 x i32>
; OzVEC: ret i32
; O1VEC2-LABEL: @enabled(
; O1VEC2: store <4 x i32>
; O1VEC2: ret i32
; OzVEC2-LABEL: @enabled(
; OzVEC2: store <4 x i32>
; OzVEC2: ret i32

define i32 @enabled(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %N
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  %1 = load i32, i32* %a, align 4
  ret i32 %1
}

; O1-LABEL: @nopragma(
; O1-NOT: store <4 x i32>
; O1: ret i32
; O2-LABEL: @nopragma(
; O2: store <4 x i32>
; O2: ret i32
; O3-LABEL: @nopragma(
; O3: store <4 x i32>
; O3: ret i32
; O3DEFAULT-LABEL: @nopragma(
; O3DEFAULT: store <4 x i32>
; O3DEFAULT: ret i32
; O3DIS-LABEL: @nopragma(
; O3DIS-NOT: store <4 x i32>
; O3DIS: ret i32
; Os-LABEL: @nopragma(
; Os: store <4 x i32>
; Os: ret i32
; Oz-LABEL: @nopragma(
; Oz-NOT: store <4 x i32>
; Oz: ret i32
; O1VEC-LABEL: @nopragma(
; O1VEC: store <4 x i32>
; O1VEC: ret i32
; OzVEC-LABEL: @nopragma(
; OzVEC: store <4 x i32>
; OzVEC: ret i32
; O1VEC2-LABEL: @nopragma(
; O1VEC2: store <4 x i32>
; O1VEC2: ret i32
; OzVEC2-LABEL: @nopragma(
; OzVEC2: store <4 x i32>
; OzVEC2: ret i32

define i32 @nopragma(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %N
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %1 = load i32, i32* %a, align 4
  ret i32 %1
}

; O1-LABEL: @disabled(
; O1-NOT: store <4 x i32>
; O1: ret i32
; O2-LABEL: @disabled(
; O2-NOT: store <4 x i32>
; O2: ret i32
; O3-LABEL: @disabled(
; O3-NOT: store <4 x i32>
; O3: ret i32
; O3DEFAULT-LABEL: @disabled(
; O3DEFAULT: store <4 x i32>
; O3DEFAULT: ret i32
; O3DIS-LABEL: @disabled(
; O3DIS-NOT: store <4 x i32>
; O3DIS: ret i32
; Os-LABEL: @disabled(
; Os-NOT: store <4 x i32>
; Os: ret i32
; Oz-LABEL: @disabled(
; Oz-NOT: store <4 x i32>
; Oz: ret i32
; O1VEC-LABEL: @disabled(
; O1VEC-NOT: store <4 x i32>
; O1VEC: ret i32
; OzVEC-LABEL: @disabled(
; OzVEC-NOT: store <4 x i32>
; OzVEC: ret i32
; O1VEC2-LABEL: @disabled(
; O1VEC2-NOT: store <4 x i32>
; O1VEC2: ret i32
; OzVEC2-LABEL: @disabled(
; OzVEC2-NOT: store <4 x i32>
; OzVEC2: ret i32

define i32 @disabled(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %N
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 48
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !2

for.end:                                          ; preds = %for.body
  %1 = load i32, i32* %a, align 4
  ret i32 %1
}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 1}
!2 = !{!2, !3}
!3 = !{!"llvm.loop.vectorize.enable", i1 0}
