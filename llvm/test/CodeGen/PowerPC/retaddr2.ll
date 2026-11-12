; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define ptr @test1() #0 {
entry:
  %0 = tail call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

; CHECK-LABEL: @test1
; CHECK: mflr {{[0-9]+}}
; CHECK: std 0, 64(1)
; CHECK-DAG: ld 3, 64(1)
; CHECK-DAG: ld [[SR:[0-9]+]], 16(1)
; CHECK: mtlr [[SR]]
; CHECK: blr

; Function Attrs: nounwind readnone
declare ptr @llvm.returnaddress(i32) #0

attributes #0 = { nounwind readnone }

