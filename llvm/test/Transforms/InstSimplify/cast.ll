; RUN: opt -S -instsimplify < %s | FileCheck %s

define i1 @test1(i1 %V) {
entry:
  %Z = zext i1 %V to i32
  %T = trunc i32 %Z to i1
  ret i1 %T
; CHECK-LABEL: define i1 @test1(
; CHECK: ret i1 %V
}

define i8* @test2(i8* %V) {
entry:
  %BC1 = bitcast i8* %V to i32*
  %BC2 = bitcast i32* %BC1 to i8*
  ret i8* %BC2
; CHECK-LABEL: define i8* @test2(
; CHECK: ret i8* %V
}

define i8* @test3(i8* %V) {
entry:
  %BC = bitcast i8* %V to i8*
  ret i8* %BC
; CHECK-LABEL: define i8* @test3(
; CHECK: ret i8* %V
}
