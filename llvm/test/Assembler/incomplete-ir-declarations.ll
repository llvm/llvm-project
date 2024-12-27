; RUN: opt -S -allow-incomplete-ir < %s | FileCheck %s

; CHECK: @fn2 = external global i8
; CHECK: @g1 = external global i8
; CHECK: @g2 = external global i8
; CHECK: @g3 = external global i8
; CHECK: @g4 = external global i8

; CHECK: declare void @fn1(i32)

define ptr @test() {
  call void @fn1(i32 0)
  call void @fn1(i32 1)
  call void @fn2(i32 2)
  call void @fn2(i32 2, i32 3)
  call void @fn2(ptr @g1)
  load i32, ptr @g2
  store i32 0, ptr @g2
  load i32, ptr @g3
  load i64, ptr @g3
  ret ptr @g4
}
