; RUN: llc -verify-machineinstrs -mcpu=ppc64 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @bar() #0 {
entry:
  tail call void @xxx([2 x i64] [i64 4607182418800017408, i64 4611686018427387904]) #0
  ret void

; CHECK-LABEL: @bar
; CHECK-DAG: li [[REG1:[0-9]+]], 1023
; CHECK-DAG: li [[REG2:[0-9]+]], {{1$}}
; CHECK-DAG: rldic 3, [[REG1]], 52, 2
; CHECK-DAG: rldic 4, [[REG2]], 62, 1
; CHECK: bl xxx
; CHECK: blr
}

declare void @xxx([2 x i64])

attributes #0 = { nounwind }

