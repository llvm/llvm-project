; RUN: opt -S -cfinserter < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-bgq-linux"

define void @test1() #0 {
entry:
  ret void

; CHECK-LABEL: define void @test1()
; CHECK: entry:
; CHECK-NEXT: call void @mcount()
; CHECK: ret void
}

define void @test2() #1 {
entry:
  ret void

; CHECK-LABEL: define void @test2()
; CHECK: entry:
; CHECK-NEXT: call void @.mcount()
; CHECK: ret void
}

attributes #0 = { "counting-function"="mcount" }
attributes #1 = { "counting-function"=".mcount" }

