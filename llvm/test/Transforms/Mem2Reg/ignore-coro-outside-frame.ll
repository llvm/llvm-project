; RUN: opt -passes=mem2reg -S -o - < %s | FileCheck %s

declare void @llvm.coro.outside.frame(ptr)

define void @test() {
; CHECK: test
; CHECK-NOT: alloca
; CHECK-NOT: call void @llvm.coro.outside.frame
  %A = alloca i32
  call void @llvm.coro.outside.frame(ptr %A)
  store i32 1, ptr %A
  ret void
}
