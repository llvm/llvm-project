; RUN: opt -S -sccp < %s | FileCheck %s

; Make sure we always consider the default edge executable for a switch
; with no cases.
declare void @foo()
define void @test1() {
; CHECK: define void @test1
; CHECK: call void @foo()
  switch i32 undef, label %d []
d:
  call void @foo()
  ret void
}
