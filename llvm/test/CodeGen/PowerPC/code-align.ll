; RUN: llc -mcpu=ppc64 < %s | FileCheck %s -check-prefix=GENERIC
; RUN: llc -mcpu=970 < %s | FileCheck %s -check-prefix=PWR
; RUN: llc -mcpu=a2 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=e500mc < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=e5500 < %s | FileCheck %s -check-prefix=BASIC
; RUN: llc -mcpu=pwr4 < %s | FileCheck %s -check-prefix=PWR
; RUN: llc -mcpu=pwr5 < %s | FileCheck %s -check-prefix=PWR
; RUN: llc -mcpu=pwr5x < %s | FileCheck %s -check-prefix=PWR
; RUN: llc -mcpu=pwr6 < %s | FileCheck %s -check-prefix=PWR
; RUN: llc -mcpu=pwr6x < %s | FileCheck %s -check-prefix=PWR
; RUN: llc -mcpu=pwr7 < %s | FileCheck %s -check-prefix=PWR
; RUN: llc -mcpu=pwr8 < %s | FileCheck %s -check-prefix=PWR
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define signext i32 @foo(i32 signext %x) #0 {
entry:
  %mul = shl nsw i32 %x, 1
  ret i32 %mul

; GENERIC-LABEL: .globl  foo
; BASIC-LABEL: .globl  foo
; PWR-LABEL: .globl  foo
; GENERIC: .p2align  2
; BASIC: .p2align  4
; PWR: .p2align  4
; GENERIC: @foo
; BASIC: @foo
; PWR: @foo
}

; Function Attrs: nounwind
define void @loop(i32 signext %x, i32* nocapture %a) #1 {
entry:
  br label %vector.body

; GENERIC-LABEL: @loop
; BASIC-LABEL: @loop
; PWR-LABEL: @loop
; GENERIC: mtctr
; BASIC: mtctr
; PWR: mtctr
; GENERIC-NOT: .p2align
; BASIC: .p2align  4
; PWR: .p2align  4
; GENERIC: lwzu
; BASIC: lwzu
; PWR: lwzu
; GENERIC: bdnz
; BASIC: bdnz
; PWR: bdnz

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %induction45 = or i64 %index, 1
  %0 = getelementptr inbounds i32, i32* %a, i64 %index
  %1 = getelementptr inbounds i32, i32* %a, i64 %induction45
  %2 = load i32, i32* %0, align 4
  %3 = load i32, i32* %1, align 4
  %4 = add nsw i32 %2, 4
  %5 = add nsw i32 %3, 4
  %6 = mul nsw i32 %4, 3
  %7 = mul nsw i32 %5, 3
  store i32 %6, i32* %0, align 4
  store i32 %7, i32* %1, align 4
  %index.next = add i64 %index, 2
  %8 = icmp eq i64 %index.next, 2048
  br i1 %8, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void
}

; Function Attrs: nounwind
define void @sloop(i32 signext %x, i32* nocapture %a) #1 {
entry:
  br label %for.body

; GENERIC-LABEL: @sloop
; BASIC-LABEL: @sloop
; PWR-LABEL: @sloop
; GENERIC: mtctr
; BASIC: mtctr
; PWR: mtctr
; GENERIC-NOT: .p2align
; BASIC: .p2align  4
; PWR: .p2align  5
; GENERIC: bdnz
; BASIC: bdnz
; PWR: bdnz

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, 4
  %mul = mul nsw i32 %add, 3
  store i32 %mul, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Function Attrs: nounwind
define void @test_minsize(i32 signext %x, i32* nocapture %a) #2 {
entry:
  br label %vector.body

; GENERIC-LABEL: @test_minsize
; BASIC-LABEL: @test_minsize
; PWR-LABEL: @test_minsize
; GENERIC: mtctr
; BASIC: mtctr
; PWR: mtctr
; GENERIC-NOT: .p2align
; BASIC-NOT: .p2align
; PWR-NOT: .p2align
; GENERIC: lwzu
; BASIC: lwzu
; PWR: lwzu
; GENERIC: bdnz
; BASIC: bdnz
; PWR: bdnz

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %induction45 = or i64 %index, 1
  %0 = getelementptr inbounds i32, i32* %a, i64 %index
  %1 = getelementptr inbounds i32, i32* %a, i64 %induction45
  %2 = load i32, i32* %0, align 4
  %3 = load i32, i32* %1, align 4
  %4 = add nsw i32 %2, 4
  %5 = add nsw i32 %3, 4
  %6 = mul nsw i32 %4, 3
  %7 = mul nsw i32 %5, 3
  store i32 %6, i32* %0, align 4
  store i32 %7, i32* %1, align 4
  %index.next = add i64 %index, 2
  %8 = icmp eq i64 %index.next, 2048
  br i1 %8, label %for.end, label %vector.body

for.end:                                          ; preds = %vector.body
  ret void
}
attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind minsize}

