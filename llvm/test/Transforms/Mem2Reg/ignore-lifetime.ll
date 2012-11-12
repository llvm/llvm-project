; RUN: opt -mem2reg -S -o - < %s | FileCheck %s

declare void @llvm.lifetime.start(i64 %size, i8* nocapture %ptr)
declare void @llvm.lifetime.end(i64 %size, i8* nocapture %ptr)

define void @test1() {
; CHECK: test1
; CHECK-NOT: alloca
  %A = alloca i32
  %B = bitcast i32* %A to i8*
  call void @llvm.lifetime.start(i64 2, i8* %B)
  store i32 1, i32* %A
  call void @llvm.lifetime.end(i64 2, i8* %B)
  ret void
}

define void @test2() {
; CHECK: test2
; CHECK-NOT: alloca
  %A = alloca {i8, i16}
  %B = getelementptr {i8, i16}* %A, i32 0, i32 0
  call void @llvm.lifetime.start(i64 2, i8* %B)
  store {i8, i16} zeroinitializer, {i8, i16}* %A
  call void @llvm.lifetime.end(i64 2, i8* %B)
  ret void
}
