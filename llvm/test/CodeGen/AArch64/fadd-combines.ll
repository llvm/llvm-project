; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: test1:
; CHECK: fadd d1, d1, d1
; CHECK: fsub d0, d0, d1
define double @test1(double %a, double %b) local_unnamed_addr #0 {
entry:
  %mul = fmul double %b, -2.000000e+00
  %add1 = fadd double %a, %mul
  ret double %add1
}

; DAGCombine will canonicalize 'a - 2.0*b' to 'a + -2.0*b'
; CHECK-LABEL: test2:
; CHECK: fadd d1, d1, d1
; CHECK: fsub d0, d0, d1
define double @test2(double %a, double %b) local_unnamed_addr #0 {
entry:
  %mul = fmul double %b, 2.000000e+00
  %add1 = fsub double %a, %mul
  ret double %add1
}

; CHECK-LABEL: test3:
; CHECK: fmul d0, d0, d1
; CHECK: fadd d1, d2, d2
; CHECK: fsub d0, d0, d1
define double @test3(double %a, double %b, double %c) local_unnamed_addr #0 {
entry:
  %mul = fmul double %a, %b
  %mul1 = fmul double %c, 2.000000e+00
  %sub = fsub double %mul, %mul1
  ret double %sub
}

; CHECK-LABEL: test4:
; CHECK: fmul d0, d0, d1
; CHECK: fadd d1, d2, d2
; CHECK: fsub d0, d0, d1
define double @test4(double %a, double %b, double %c) local_unnamed_addr #0 {
entry:
  %mul = fmul double %a, %b
  %mul1 = fmul double %c, -2.000000e+00
  %add2 = fadd double %mul, %mul1
  ret double %add2
}

; CHECK-LABEL: test5:
; CHECK: fadd v1.4s, v1.4s, v1.4s
; CHECK: fsub v0.4s, v0.4s, v1.4s
define <4 x float> @test5(<4 x float> %a, <4 x float> %b) {
  %mul = fmul <4 x float> %b, <float -2.0, float -2.0, float -2.0, float -2.0>
  %add = fadd <4 x float> %a, %mul
  ret <4 x float> %add
}

; CHECK-LABEL: test6:
; CHECK: fadd v1.4s, v1.4s, v1.4s
; CHECK: fsub v0.4s, v0.4s, v1.4s
define <4 x float> @test6(<4 x float> %a, <4 x float> %b) {
  %mul = fmul <4 x float> %b, <float 2.0, float 2.0, float 2.0, float 2.0>
  %add = fsub <4 x float> %a, %mul
  ret <4 x float> %add
}

; Don't fold (fadd A, (fmul B, -2.0)) -> (fsub A, (fadd B, B)) if the fmul has
; multiple uses.
; CHECK-LABEL: test7:
; CHECK: fmul
define double @test7(double %a, double %b) local_unnamed_addr #0 {
entry:
  %mul = fmul double %b, -2.000000e+00
  %add1 = fadd double %a, %mul
  call void @use(double %mul)
  ret double %add1
}

declare void @use(double)
