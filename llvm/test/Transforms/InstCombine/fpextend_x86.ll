; RUN: opt < %s -instcombine -mtriple=x86_64-apple-macosx -S | FileCheck %s
target triple = "x86_64-apple-macosx"

define double @test1(double %a, double %b) nounwind {
  %wa = fpext double %a to x86_fp80
  %wb = fpext double %b to x86_fp80
  %wr = fadd x86_fp80 %wa, %wb
  %r = fptrunc x86_fp80 %wr to double
  ret double %r
; CHECK: test1
; CHECK: fadd x86_fp80
; CHECK: ret
}

define double @test2(double %a, double %b) nounwind {
  %wa = fpext double %a to x86_fp80
  %wb = fpext double %b to x86_fp80
  %wr = fsub x86_fp80 %wa, %wb
  %r = fptrunc x86_fp80 %wr to double
  ret double %r
; CHECK: test2
; CHECK: fsub x86_fp80
; CHECK: ret
}

define double @test3(double %a, double %b) nounwind {
  %wa = fpext double %a to x86_fp80
  %wb = fpext double %b to x86_fp80
  %wr = fmul x86_fp80 %wa, %wb
  %r = fptrunc x86_fp80 %wr to double
  ret double %r
; CHECK: test3
; CHECK: fmul x86_fp80
; CHECK: ret
}

define double @test4(double %a, half %b) nounwind {
  %wa = fpext double %a to x86_fp80
  %wb = fpext half %b to x86_fp80
  %wr = fmul x86_fp80 %wa, %wb
  %r = fptrunc x86_fp80 %wr to double
  ret double %r
; CHECK: test4
; CHECK: fmul double
; CHECK: ret
}

define double @test5(double %a, double %b) nounwind {
  %wa = fpext double %a to x86_fp80
  %wb = fpext double %b to x86_fp80
  %wr = fdiv x86_fp80 %wa, %wb
  %r = fptrunc x86_fp80 %wr to double
  ret double %r
; CHECK: test5
; CHECK: fdiv x86_fp80
; CHECK: ret
}
