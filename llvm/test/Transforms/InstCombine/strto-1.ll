; Test that the strto* library call simplifiers works correctly.
;
; RUN: opt < %s -instcombine -inferattrs -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i32 @strtol(ptr %s, ptr %endptr, i32 %base)
; CHECK: declare i32 @strtol(ptr readonly, ptr nocapture, i32)

declare double @strtod(ptr %s, ptr %endptr)
; CHECK: declare double @strtod(ptr readonly, ptr nocapture)

declare float @strtof(ptr %s, ptr %endptr)
; CHECK: declare float @strtof(ptr readonly, ptr nocapture)

declare i64 @strtoul(ptr %s, ptr %endptr, i32 %base)
; CHECK: declare i64 @strtoul(ptr readonly, ptr nocapture, i32)

declare i64 @strtoll(ptr %s, ptr %endptr, i32 %base)
; CHECK: declare i64 @strtoll(ptr readonly, ptr nocapture, i32)

declare double @strtold(ptr %s, ptr %endptr)
; CHECK: declare double @strtold(ptr readonly, ptr nocapture)

declare i64 @strtoull(ptr %s, ptr %endptr, i32 %base)
; CHECK: declare i64 @strtoull(ptr readonly, ptr nocapture, i32)

define void @test_simplify1(ptr %x, ptr %endptr) {
; CHECK-LABEL: @test_simplify1(
  call i32 @strtol(ptr %x, ptr null, i32 10)
; CHECK-NEXT: call i32 @strtol(ptr nocapture %x, ptr null, i32 10)
  ret void
}

define void @test_simplify2(ptr %x, ptr %endptr) {
; CHECK-LABEL: @test_simplify2(
  call double @strtod(ptr %x, ptr null)
; CHECK-NEXT: call double @strtod(ptr nocapture %x, ptr null)
  ret void
}

define void @test_simplify3(ptr %x, ptr %endptr) {
; CHECK-LABEL: @test_simplify3(
  call float @strtof(ptr %x, ptr null)
; CHECK-NEXT: call float @strtof(ptr nocapture %x, ptr null)
  ret void
}

define void @test_simplify4(ptr %x, ptr %endptr) {
; CHECK-LABEL: @test_simplify4(
  call i64 @strtoul(ptr %x, ptr null, i32 10)
; CHECK-NEXT: call i64 @strtoul(ptr nocapture %x, ptr null, i32 10)
  ret void
}

define void @test_simplify5(ptr %x, ptr %endptr) {
; CHECK-LABEL: @test_simplify5(
  call i64 @strtoll(ptr %x, ptr null, i32 10)
; CHECK-NEXT: call i64 @strtoll(ptr nocapture %x, ptr null, i32 10)
  ret void
}

define void @test_simplify6(ptr %x, ptr %endptr) {
; CHECK-LABEL: @test_simplify6(
  call double @strtold(ptr %x, ptr null)
; CHECK-NEXT: call double @strtold(ptr nocapture %x, ptr null)
  ret void
}

define void @test_simplify7(ptr %x, ptr %endptr) {
; CHECK-LABEL: @test_simplify7(
  call i64 @strtoull(ptr %x, ptr null, i32 10)
; CHECK-NEXT: call i64 @strtoull(ptr nocapture %x, ptr null, i32 10)
  ret void
}

define void @test_no_simplify1(ptr %x, ptr %endptr) {
; CHECK-LABEL: @test_no_simplify1(
  call i32 @strtol(ptr %x, ptr %endptr, i32 10)
; CHECK-NEXT: call i32 @strtol(ptr %x, ptr %endptr, i32 10)
  ret void
}
