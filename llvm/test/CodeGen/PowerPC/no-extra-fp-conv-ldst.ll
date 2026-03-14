; RUN: llc -verify-machineinstrs -mcpu=a2 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readonly
define double @test1(ptr nocapture readonly %x) #0 {
entry:
  %0 = load i64, ptr %x, align 8
  %conv = sitofp i64 %0 to double
  ret double %conv

; CHECK-LABEL: @test1
; CHECK: lfd [[REG1:[0-9]+]], 0(3)
; CHECK: fcfid 1, [[REG1]]
; CHECK: blr
}

; Function Attrs: nounwind readonly
define double @test2(ptr nocapture readonly %x) #0 {
entry:
  %0 = load i32, ptr %x, align 4
  %conv = sitofp i32 %0 to double
  ret double %conv

; CHECK-LABEL: @test2
; CHECK: lfiwax [[REG1:[0-9]+]], 0, 3
; CHECK: fcfid 1, [[REG1]]
; CHECK: blr
}

; Function Attrs: nounwind readnone
define float @foo(float %X) #0 {
entry:
  %conv = fptosi float %X to i32
  %conv1 = sitofp i32 %conv to float
  ret float %conv1

; CHECK-LABEL: @foo
; CHECK: friz 1, 1
; CHECK: blr
}

; Function Attrs: nounwind readnone
define double @food(double %X) #0 {
entry:
  %conv = fptosi double %X to i32
  %conv1 = sitofp i32 %conv to double
  ret double %conv1

; CHECK-LABEL: @food
; CHECK: friz 1, 1
; CHECK: blr
}

; Function Attrs: nounwind readnone
define float @foou(float %X) #0 {
entry:
  %conv = fptoui float %X to i32
  %conv1 = uitofp i32 %conv to float
  ret float %conv1

; CHECK-LABEL: @foou
; CHECK: friz 1, 1
; CHECK: blr
}

; Function Attrs: nounwind readnone
define double @fooud(double %X) #0 {
entry:
  %conv = fptoui double %X to i32
  %conv1 = uitofp i32 %conv to double
  ret double %conv1

; CHECK-LABEL: @fooud
; CHECK: friz 1, 1
; CHECK: blr
}

attributes #0 = { nounwind readonly "no-signed-zeros-fp-math"="true" }

