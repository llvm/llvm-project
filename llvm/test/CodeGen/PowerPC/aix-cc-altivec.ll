; RUN: not --crash llc < %s -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr8 2>&1 | FileCheck %s

; This test expects a compiler diagnostic for an AIX limitation on Altivec
; support.  When the Altivec limitation diagnostic is removed, this test
; should compile clean and fail in order to alert the author to validate the
; instructions emitted to initialize the GPR for the double vararg.
; The mfvsrwz and mfvsrd instructions should be used to initialize the GPR for
; the double vararg without going through memory.

@f1 = global float 0.000000e+00, align 4

define void @call_test_vararg() {
entry:
  %0 = load float, float* @f1, align 4
  %conv = fpext float %0 to double
  call void (i32, ...) @test_vararg(i32 42, double %conv, float %0)
  ret void
}

declare void @test_vararg(i32, ...)

; CHECK: LLVM ERROR: Altivec support is unimplemented on AIX.
