; RUN: opt -S -dse < %s | FileCheck %s

declare void @f()
declare noalias i8* @malloc(i32) nounwind

define void @test_0() {
; CHECK-LABEL: @test_0(
  %m = call i8* @malloc(i32 24)
  tail call void @f() [ "unknown"(i8* %m) ]
; CHECK: store i8 -19, i8* %m
  store i8 -19, i8* %m
  ret void
}

define i8* @test_1() {
; CHECK-LABEL: @test_1(
  %m = call i8* @malloc(i32 24)
  tail call void @f() [ "unknown"(i8* %m) ]
  store i8 -19, i8* %m
  tail call void @f()
  store i8 101, i8* %m

; CHECK: tail call void @f() [ "unknown"(i8* %m) ]
; CHECK: store i8 -19, i8* %m
; CHECK: tail call void @f()
; CHECK: store i8 101, i8* %m

  ret i8* %m
}
