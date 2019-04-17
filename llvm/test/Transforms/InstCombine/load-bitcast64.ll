; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "p:64:64:64"


define i64* @test1(i8* %x) {
entry:
; CHECK-LABEL: @test1(
; CHECK: load i64*, i64**
; CHECK: ret
  %a = bitcast i8* %x to i64*
  %b = load i64, i64* %a
  %c = inttoptr i64 %b to i64*

  ret i64* %c
}

define i32* @test2(i8* %x) {
entry:
; CHECK-LABEL: @test2(
; CHECK: load i32, i32*
; CHECK: ret
  %a = bitcast i8* %x to i32*
  %b = load i32, i32* %a
  %c = inttoptr i32 %b to i32*

  ret i32* %c
}

define i64* @test3(i8* %x) {
entry:
; CHECK-LABEL: @test3(
; CHECK: load i32, i32*
; CHECK: ret
  %a = bitcast i8* %x to i32*
  %b = load i32, i32* %a
  %c = inttoptr i32 %b to i64*

  ret i64* %c
}

define i64 @test4(i8* %x) {
entry:
; CHECK-LABEL: @test4(
; CHECK: load i64, i64*
; CHECK: ret
  %a = bitcast i8* %x to i64**
  %b = load i64*, i64** %a
  %c = ptrtoint i64* %b to i64

  ret i64 %c
}

define i32 @test5(i8* %x) {
entry:
; CHECK-LABEL: @test5(
; CHECK: load i64, i64*
; CHECK: trunc
; CHECK: ret
  %a = bitcast i8* %x to i32**
  %b = load i32*, i32** %a
  %c = ptrtoint i32* %b to i32

  ret i32 %c
}

define i64 @test6(i8* %x) {
entry:
; CHECK-LABEL: @test6(
; CHECK: load i64, i64*
; CHECK: ret
  %a = bitcast i8* %x to i32**
  %b = load i32*, i32** %a
  %c = ptrtoint i32* %b to i64

  ret i64 %c
}

