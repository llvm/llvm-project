; RUN: opt -S -early-cse < %s | FileCheck %s

@var = global i32 undef
declare void @foo() nounwind

define void @test() {
; CHECK-LABEL: @test(
; CHECK-NOT: store
  store i32 1, i32* @var
; CHECK: call void @foo()
  call void @foo() writeonly
; CHECK: store i32 2, i32* @var
  store i32 2, i32* @var
  ret void
}
