; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-bgq-linux"

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: @test1
; CHECK: bl mcount
; CHECK-NOT: mcount
; CHECK: blr
}

attributes #0 = { "counting-function"="mcount" }

