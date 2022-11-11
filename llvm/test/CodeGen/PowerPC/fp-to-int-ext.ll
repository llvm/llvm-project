; RUN: llc -verify-machineinstrs -mcpu=a2 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define double @foo1(ptr %x) #0 {
entry:
  %0 = load i32, ptr %x, align 4
  %conv = sext i32 %0 to i64
  %conv1 = sitofp i64 %conv to double
  ret double %conv1

; CHECK-LABEL: @foo1
; CHECK: lfiwax [[REG1:[0-9]+]], 0, 3
; CHECK: fcfid 1, [[REG1]]
; CHECK: blr
}

define double @foo2(ptr %x) #0 {
entry:
  %0 = load i32, ptr %x, align 4
  %conv = zext i32 %0 to i64
  %conv1 = sitofp i64 %conv to double
  ret double %conv1

; CHECK-LABEL: @foo2
; CHECK: lfiwzx [[REG1:[0-9]+]], 0, 3
; CHECK: fcfid 1, [[REG1]]
; CHECK: blr
}

define double @foo3(ptr %x) #0 {
entry:
  %0 = load i32, ptr %x, align 4
  %1 = add i32 %0, 8
  %conv = zext i32 %1 to i64
  %conv1 = sitofp i64 %conv to double
  ret double %conv1

; CHECK-LABEL: @foo3
; CHECK-DAG: lwz [[REG1:[0-9]+]], 0(3)
; CHECK-DAG: addi [[REG3:[0-9]+]], 1,
; CHECK-DAG: addi [[REG2:[0-9]+]], [[REG1]], 8
; CHECK-DAG: stw [[REG2]],
; CHECK: lfiwzx [[REG4:[0-9]+]], 0, [[REG3]]
; CHECK: fcfid 1, [[REG4]]
; CHECK: blr
}

define double @foo4(ptr %x) #0 {
entry:
  %0 = load i32, ptr %x, align 4
  %1 = add i32 %0, 8
  %conv = sext i32 %1 to i64
  %conv1 = sitofp i64 %conv to double
  ret double %conv1

; CHECK-LABEL: @foo4
; CHECK-DAG: lwz [[REG1:[0-9]+]], 0(3)
; CHECK-DAG: addi [[REG3:[0-9]+]], 1,
; CHECK-DAG: addi [[REG2:[0-9]+]], [[REG1]], 8
; CHECK-DAG: stw [[REG2]],
; CHECK: lfiwax [[REG4:[0-9]+]], 0, [[REG3]]
; CHECK: fcfid 1, [[REG4]]
; CHECK: blr
}

attributes #0 = { nounwind }

