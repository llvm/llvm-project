; RUN: opt -passes=mem2reg -S -o - < %s | FileCheck %s

declare void @llvm.lifetime.start.p0(i64 %size, ptr nocapture %ptr)
declare void @llvm.lifetime.end.p0(i64 %size, ptr nocapture %ptr)

define void @test1() {
; CHECK: test1
; CHECK-NOT: alloca
  %A = alloca i32
  call void @llvm.lifetime.start.p0(i64 2, ptr %A)
  store i32 1, ptr %A
  call void @llvm.lifetime.end.p0(i64 2, ptr %A)
  ret void
}

define void @test2() {
; CHECK: test2
; CHECK-NOT: alloca
  %A = alloca {i8, i16}
  %B = getelementptr {i8, i16}, ptr %A, i32 0, i32 0
  call void @llvm.lifetime.start.p0(i64 2, ptr %B)
  store {i8, i16} zeroinitializer, ptr %A
  call void @llvm.lifetime.end.p0(i64 2, ptr %B)
  ret void
}
