; RUN: opt -S -inline < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1() {
entry:
  call void @test2()
  ret void
}

define internal void @test2() {
entry:
  call void undef()
  ret void
}

; CHECK-LABEL: define void @test1(
; CHECK: call void undef(
; CHECK: ret void
