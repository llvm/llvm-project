; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
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
; CHECK-DAG: cmpw
; CHECK-DAG: li [[REG1:[0-9]+]], 0
; CHECK-DAG: li [[REG2:[0-9]+]], 16
; CHECK: isel 3, [[REG2]], [[REG1]],
; CHECK: blr
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
; CHECK-DAG: cmpw
; CHECK-DAG: li [[REG1:[0-9]+]], 5
; CHECK-DAG: li [[REG2:[0-9]+]], 21
; CHECK: isel 3, [[REG2]], [[REG1]],
; CHECK: blr
}

; Function Attrs: nounwind readnone
define signext i32 @foo3(i32 signext %a, i32 signext %b) #0 {
entry:
  %cmp = icmp sle i32 %a, %b
  %conv = zext i1 %cmp to i32
  %shl = shl nuw nsw i32 %conv, 4
  ret i32 %shl

; CHECK-LABEL: @foo3
; CHECK-DAG: cmpw
; CHECK-DAG: li [[REG1:[0-9]+]], 16
; CHECK: isel 3, 0, [[REG1]],
; CHECK: blr
}

attributes #0 = { nounwind readnone }

