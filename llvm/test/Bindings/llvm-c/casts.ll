; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Test cast instructions for echo command coverage.

define i8 @test_trunc(i32 %x) {
  %result = trunc i32 %x to i8
  ret i8 %result
}

define i64 @test_sext(i32 %x) {
  %result = sext i32 %x to i64
  ret i64 %result
}

define float @test_fptrunc(double %x) {
  %result = fptrunc double %x to float
  ret float %result
}

define double @test_fpext(float %x) {
  %result = fpext float %x to double
  ret double %result
}

define i32 @test_fptosi(float %x) {
  %result = fptosi float %x to i32
  ret i32 %result
}

define i32 @test_fptoui(float %x) {
  %result = fptoui float %x to i32
  ret i32 %result
}

define float @test_sitofp(i32 %x) {
  %result = sitofp i32 %x to float
  ret float %result
}

define float @test_uitofp(i32 %x) {
  %result = uitofp i32 %x to float
  ret float %result
}

define i64 @test_ptrtoint(ptr %p) {
  %result = ptrtoint ptr %p to i64
  ret i64 %result
}

define ptr @test_inttoptr(i64 %x) {
  %result = inttoptr i64 %x to ptr
  ret ptr %result
}
