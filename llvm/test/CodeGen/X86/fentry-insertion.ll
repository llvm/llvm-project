; RUN: llc %s -o - | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: @test1
; CHECK: callq __fentry__
; CHECK-NOT: mcount
; CHECK: retq
}

attributes #0 = { "fentry-call"="true" }

