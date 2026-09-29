; RUN: opt -S -passes=instcombine < %s | FileCheck %s

define void @test1() {
entry:
  call void @tan()
  ret void
}
; CHECK-LABEL: define void @test1(
; CHECK:      call void @tan()
; CHECK-NEXT: ret void

declare void @tan()
