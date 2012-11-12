; Test that the strto* library call simplifiers works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i64 @strtol(i8* %s, i8** %endptr, i32 %base)
; CHECK: declare i64 @strtol(i8*, i8**, i32)

declare double @strtod(i8* %s, i8** %endptr, i32 %base)
; CHECK: declare double @strtod(i8*, i8**, i32)

declare float @strtof(i8* %s, i8** %endptr, i32 %base)
; CHECK: declare float @strtof(i8*, i8**, i32)

declare i64 @strtoul(i8* %s, i8** %endptr, i32 %base)
; CHECK: declare i64 @strtoul(i8*, i8**, i32)

declare i64 @strtoll(i8* %s, i8** %endptr, i32 %base)
; CHECK: declare i64 @strtoll(i8*, i8**, i32)

declare double @strtold(i8* %s, i8** %endptr)
; CHECK: declare double @strtold(i8*, i8**)

declare i64 @strtoull(i8* %s, i8** %endptr, i32 %base)
; CHECK: declare i64 @strtoull(i8*, i8**, i32)

define void @test_simplify1(i8* %x, i8** %endptr) {
; CHECK: @test_simplify1
  call i64 @strtol(i8* %x, i8** null, i32 10)
; CHECK-NEXT: call i64 @strtol(i8* nocapture %x, i8** null, i32 10)
  ret void
}

define void @test_simplify2(i8* %x, i8** %endptr) {
; CHECK: @test_simplify2
  call double @strtod(i8* %x, i8** null, i32 10)
; CHECK-NEXT: call double @strtod(i8* nocapture %x, i8** null, i32 10)
  ret void
}

define void @test_simplify3(i8* %x, i8** %endptr) {
; CHECK: @test_simplify3
  call float @strtof(i8* %x, i8** null, i32 10)
; CHECK-NEXT: call float @strtof(i8* nocapture %x, i8** null, i32 10)
  ret void
}

define void @test_simplify4(i8* %x, i8** %endptr) {
; CHECK: @test_simplify4
  call i64 @strtoul(i8* %x, i8** null, i32 10)
; CHECK-NEXT: call i64 @strtoul(i8* nocapture %x, i8** null, i32 10)
  ret void
}

define void @test_simplify5(i8* %x, i8** %endptr) {
; CHECK: @test_simplify5
  call i64 @strtoll(i8* %x, i8** null, i32 10)
; CHECK-NEXT: call i64 @strtoll(i8* nocapture %x, i8** null, i32 10)
  ret void
}

define void @test_simplify6(i8* %x, i8** %endptr) {
; CHECK: @test_simplify6
  call double @strtold(i8* %x, i8** null)
; CHECK-NEXT: call double @strtold(i8* nocapture %x, i8** null)
  ret void
}

define void @test_simplify7(i8* %x, i8** %endptr) {
; CHECK: @test_simplify7
  call i64 @strtoull(i8* %x, i8** null, i32 10)
; CHECK-NEXT: call i64 @strtoull(i8* nocapture %x, i8** null, i32 10)
  ret void
}

define void @test_no_simplify1(i8* %x, i8** %endptr) {
; CHECK: @test_no_simplify1
  call i64 @strtol(i8* %x, i8** %endptr, i32 10)
; CHECK-NEXT: call i64 @strtol(i8* %x, i8** %endptr, i32 10)
  ret void
}
