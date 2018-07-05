; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -ppc-gen-isel=false < %s | FileCheck --check-prefix=CHECK-NO-ISEL %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define signext i32 @foo(i32 signext %a, i32 signext %b) #0 {
entry:
  %cmp = icmp slt i32 %a, %b
  %conv = zext i1 %cmp to i32
  %shl = shl nuw nsw i32 %conv, 4
  ret i32 %shl

; CHECK-LABEL: @foo
; CHECK-NO-ISEL-LABEL: @foo
; CHECK-DAG: cmpw
; CHECK-DAG: li [[REG1:[0-9]+]], 0
; CHECK-DAG: li [[REG2:[0-9]+]], 16
; CHECK: isel 3, [[REG2]], [[REG1]],
; CHECK: blr

; CHECK-NO-ISEL: bclr 12, 0, 0
; CHECK-NO-ISEL: ori 3, 5, 0
; CHECK-NO-ISEL-NEXT: blr
}

; Function Attrs: nounwind readnone
define signext i32 @foo2(i32 signext %a, i32 signext %b) #0 {
entry:
  %cmp = icmp slt i32 %a, %b
  %conv = zext i1 %cmp to i32
  %shl = shl nuw nsw i32 %conv, 4
  %add1 = or i32 %shl, 5
  ret i32 %add1

; CHECK-LABEL: @foo2
; CHECK-NO-ISEL-LABEL: @foo2
; CHECK-DAG: cmpw
; CHECK-DAG: li [[REG1:[0-9]+]], 5
; CHECK-DAG: li [[REG2:[0-9]+]], 21
; CHECK: isel 3, [[REG2]], [[REG1]],
; CHECK: blr

; CHECK-NO-ISEL: bclr 12, 0, 0
; CHECK-NO-ISEL: ori 3, 5, 0
; CHECK-NO-ISEL-NEXT: blr
}

; Function Attrs: nounwind readnone
define signext i32 @foo3(i32 signext %a, i32 signext %b) #0 {
entry:
  %cmp = icmp sle i32 %a, %b
  %conv = zext i1 %cmp to i32
  %shl = shl nuw nsw i32 %conv, 4
  ret i32 %shl

; CHECK-LABEL: @foo3
; CHECK-NO-ISEL-LABEL: @foo3
; CHECK-DAG: cmpw
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK: isel 3, 0, [[REG1]],
; CHECK: blr

; CHECK-NO-ISEL: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 3, 5, 0
; CHECK-NO-ISEL-NEXT: blr
; CHECK-NO-ISEL-NEXT: [[TRUE]]
; CHECK-NO-ISEL-NEXT: addi 3, 0, 0
; CHECK-NO-ISEL-NEXT: blr
}

attributes #0 = { nounwind readnone }

