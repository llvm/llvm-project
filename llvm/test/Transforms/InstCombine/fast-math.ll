; RUN: opt < %s -instcombine -S | FileCheck %s

; testing-case "float fold(float a) { return 1.2f * a * 2.3f; }"
; 1.2f and 2.3f is supposed to be fold.
define float @fold(float %a) {
  %mul = fmul fast float %a, 0x3FF3333340000000
  %mul1 = fmul fast float %mul, 0x4002666660000000
  ret float %mul1
; CHECK-LABEL: @fold(
; CHECK: fmul fast float %a, 0x4006147AE0000000
}

; Same testing-case as the one used in fold() except that the operators have
; fixed FP mode.
define float @notfold(float %a) {
; CHECK-LABEL: @notfold(
; CHECK: %mul = fmul fast float %a, 0x3FF3333340000000
  %mul = fmul fast float %a, 0x3FF3333340000000
  %mul1 = fmul float %mul, 0x4002666660000000
  ret float %mul1
}

define float @fold2(float %a) {
; CHECK-LABEL: @fold2(
; CHECK: fmul fast float %a, 0x4006147AE0000000
  %mul = fmul float %a, 0x3FF3333340000000
  %mul1 = fmul fast float %mul, 0x4002666660000000
  ret float %mul1
}

; C * f1 + f1 = (C+1) * f1
define double @fold3(double %f1) {
  %t1 = fmul fast double 2.000000e+00, %f1
  %t2 = fadd fast double %f1, %t1
  ret double %t2
; CHECK-LABEL: @fold3(
; CHECK: fmul fast double %f1, 3.000000e+00
}

; (C1 - X) + (C2 - Y) => (C1+C2) - (X + Y)
define float @fold4(float %f1, float %f2) {
  %sub = fsub float 4.000000e+00, %f1
  %sub1 = fsub float 5.000000e+00, %f2
  %add = fadd fast float %sub, %sub1
  ret float %add
; CHECK-LABEL: @fold4(
; CHECK: %1 = fadd fast float %f1, %f2
; CHECK: fsub fast float 9.000000e+00, %1
}

; (X + C1) + C2 => X + (C1 + C2)
define float @fold5(float %f1, float %f2) {
  %add = fadd float %f1, 4.000000e+00
  %add1 = fadd fast float %add, 5.000000e+00
  ret float %add1
; CHECK-LABEL: @fold5(
; CHECK: fadd fast float %f1, 9.000000e+00
}

; (X + X) + X => 3.0 * X
define float @fold6(float %f1) {
  %t1 = fadd fast float %f1, %f1
  %t2 = fadd fast float %f1, %t1
  ret float %t2
; CHECK-LABEL: @fold6(
; CHECK: fmul fast float %f1, 3.000000e+00
}

; C1 * X + (X + X) = (C1 + 2) * X
define float @fold7(float %f1) {
  %t1 = fmul fast float %f1, 5.000000e+00
  %t2 = fadd fast float %f1, %f1
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fold7(
; CHECK: fmul fast float %f1, 7.000000e+00
}

; (X + X) + (X + X) => 4.0 * X
define float @fold8(float %f1) {
  %t1 = fadd fast float %f1, %f1
  %t2 = fadd fast float %f1, %f1
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fold8
; CHECK: fmul fast float %f1, 4.000000e+00
}

; X - (X + Y) => 0 - Y
define float @fold9(float %f1, float %f2) {
  %t1 = fadd float %f1, %f2
  %t3 = fsub fast float %f1, %t1
  ret float %t3

; CHECK-LABEL: @fold9(
; CHECK: fsub fast float -0.000000e+00, %f2
}

; Let C3 = C1 + C2. (f1 + C1) + (f2 + C2) => (f1 + f2) + C3 instead of
; "(f1 + C3) + f2" or "(f2 + C3) + f1". Placing constant-addend at the
; top of resulting simplified expression tree may potentially reveal some
; optimization opportunities in the super-expression trees.
;
define float @fold10(float %f1, float %f2) {
  %t1 = fadd fast float 2.000000e+00, %f1
  %t2 = fsub fast float %f2, 3.000000e+00
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fold10(
; CHECK: %t3 = fadd fast float %t2, -1.000000e+00
; CHECK: ret float %t3
}

; once cause Crash/miscompilation
define float @fail1(float %f1, float %f2) {
  %conv3 = fadd fast float %f1, -1.000000e+00
  %add = fadd fast float %conv3, %conv3
  %add2 = fadd fast float %add, %conv3
  ret float %add2
; CHECK-LABEL: @fail1(
; CHECK: ret
}

define double @fail2(double %f1, double %f2) {
  %t1 = fsub fast double %f1, %f2
  %t2 = fadd fast double %f1, %f2
  %t3 = fsub fast double %t1, %t2
  ret double %t3
; CHECK-LABEL: @fail2(
; CHECK: ret
}

; c1 * x - x => (c1 - 1.0) * x
define float @fold13(float %x) {
  %mul = fmul fast float %x, 7.000000e+00
  %sub = fsub fast float %mul, %x
  ret float %sub
; CHECK: fold13
; CHECK: fmul fast float %x, 6.000000e+00
; CHECK: ret
}

; -x + y => y - x
define float @fold14(float %x, float %y) {
  %neg = fsub fast float -0.0, %x
  %add = fadd fast float %neg, %y
  ret float %add
; CHECK: fold14
; CHECK: fsub fast float %y, %x
; CHECK: ret
}

; x + -y => x - y
define float @fold15(float %x, float %y) {
  %neg = fsub fast float -0.0, %y
  %add = fadd fast float %x, %neg
  ret float %add
; CHECK: fold15
; CHECK: fsub fast float %x, %y
; CHECK: ret
}

; (select X+Y, X-Y) => X + (select Y, -Y)
define float @fold16(float %x, float %y) {
  %cmp = fcmp ogt float %x, %y
  %plus = fadd fast float %x, %y
  %minus = fsub fast float %x, %y
  %r = select i1 %cmp, float %plus, float %minus
  ret float %r
; CHECK: fold16
; CHECK: fsub fast float
; CHECK: select
; CHECK: fadd fast float
; CHECK: ret
}



; =========================================================================
;
;   Testing-cases about fmul begin
;
; =========================================================================

; ((X*C1) + C2) * C3 => (X * (C1*C3)) + (C2*C3) (i.e. distribution)
define float @fmul_distribute1(float %f1) {
  %t1 = fmul float %f1, 6.0e+3
  %t2 = fadd float %t1, 2.0e+3
  %t3 = fmul fast float %t2, 5.0e+3
  ret float %t3
; CHECK-LABEL: @fmul_distribute1(
; CHECK: %1 = fmul fast float %f1, 3.000000e+07
; CHECK: %t3 = fadd fast float %1, 1.000000e+07
}

; (X/C1 + C2) * C3 => X/(C1/C3) + C2*C3
define double @fmul_distribute2(double %f1, double %f2) {
  %t1 = fdiv double %f1, 3.0e+0
  %t2 = fadd double %t1, 5.0e+1
  ; 0x10000000000000 = DBL_MIN
  %t3 = fmul fast double %t2, 0x10000000000000
  ret double %t3

; CHECK-LABEL: @fmul_distribute2(
; CHECK: %1 = fdiv fast double %f1, 0x7FE8000000000000
; CHECK: fadd fast double %1, 0x69000000000000
}

; 5.0e-1 * DBL_MIN yields denormal, so "(f1*3.0 + 5.0e-1) * DBL_MIN" cannot
; be simplified into f1 * (3.0*DBL_MIN) + (5.0e-1*DBL_MIN)
define double @fmul_distribute3(double %f1) {
  %t1 = fdiv double %f1, 3.0e+0
  %t2 = fadd double %t1, 5.0e-1
  %t3 = fmul fast double %t2, 0x10000000000000
  ret double %t3

; CHECK-LABEL: @fmul_distribute3(
; CHECK: fmul fast double %t2, 0x10000000000000
}

; ((X*C1) + C2) * C3 => (X * (C1*C3)) + (C2*C3) (i.e. distribution)
define float @fmul_distribute4(float %f1) {
  %t1 = fmul float %f1, 6.0e+3
  %t2 = fsub float 2.0e+3, %t1
  %t3 = fmul fast float %t2, 5.0e+3
  ret float %t3
; CHECK-LABEL: @fmul_distribute4(
; CHECK: %1 = fmul fast float %f1, 3.000000e+07
; CHECK: %t3 = fsub fast float 1.000000e+07, %1
}

; C1/X * C2 => (C1*C2) / X
define float @fmul2(float %f1) {
  %t1 = fdiv float 2.0e+3, %f1
  %t3 = fmul fast float %t1, 6.0e+3
  ret float %t3
; CHECK-LABEL: @fmul2(
; CHECK: fdiv fast float 1.200000e+07, %f1
}

; X/C1 * C2 => X * (C2/C1) is disabled if X/C1 has multiple uses
@fmul2_external = external global float
define float @fmul2_disable(float %f1) {
  %div = fdiv fast float 1.000000e+00, %f1
  store float %div, float* @fmul2_external
  %mul = fmul fast float %div, 2.000000e+00
  ret float %mul
; CHECK-LABEL: @fmul2_disable
; CHECK: store
; CHECK: fmul fast
}

; X/C1 * C2 => X * (C2/C1) (if C2/C1 is normal Fp)
define float @fmul3(float %f1, float %f2) {
  %t1 = fdiv float %f1, 2.0e+3
  %t3 = fmul fast float %t1, 6.0e+3
  ret float %t3
; CHECK-LABEL: @fmul3(
; CHECK: fmul fast float %f1, 3.000000e+00
}

define <4 x float> @fmul3_vec(<4 x float> %f1, <4 x float> %f2) {
  %t1 = fdiv <4 x float> %f1, <float 2.0e+3, float 3.0e+3, float 2.0e+3, float 1.0e+3>
  %t3 = fmul fast <4 x float> %t1, <float 6.0e+3, float 6.0e+3, float 2.0e+3, float 1.0e+3>
  ret <4 x float> %t3
; CHECK-LABEL: @fmul3_vec(
; CHECK: fmul fast <4 x float> %f1, <float 3.000000e+00, float 2.000000e+00, float 1.000000e+00, float 1.000000e+00>
}

; Make sure fmul with constant expression doesn't assert.
define <4 x float> @fmul3_vec_constexpr(<4 x float> %f1, <4 x float> %f2) {
  %constExprMul = bitcast i128 trunc (i160 bitcast (<5 x float> <float 6.0e+3, float 6.0e+3, float 2.0e+3, float 1.0e+3, float undef> to i160) to i128) to <4 x float>  
  %t1 = fdiv <4 x float> %f1, <float 2.0e+3, float 3.0e+3, float 2.0e+3, float 1.0e+3>
  %t3 = fmul fast <4 x float> %t1, %constExprMul
  ret <4 x float> %t3
}

; Rule "X/C1 * C2 => X * (C2/C1) is not applicable if C2/C1 is either a special
; value of a denormal. The 0x3810000000000000 here take value FLT_MIN
;
define float @fmul4(float %f1, float %f2) {
  %t1 = fdiv float %f1, 2.0e+3
  %t3 = fmul fast float %t1, 0x3810000000000000
  ret float %t3
; CHECK-LABEL: @fmul4(
; CHECK: fmul fast float %t1, 0x3810000000000000
}

; X / C1 * C2 => X / (C2/C1) if  C1/C2 is either a special value of a denormal,
;  and C2/C1 is a normal value.
;
define float @fmul5(float %f1, float %f2) {
  %t1 = fdiv float %f1, 3.0e+0
  %t3 = fmul fast float %t1, 0x3810000000000000
  ret float %t3
; CHECK-LABEL: @fmul5(
; CHECK: fdiv fast float %f1, 0x47E8000000000000
}

; (X*Y) * X => (X*X) * Y
define float @fmul6(float %f1, float %f2) {
  %mul = fmul float %f1, %f2
  %mul1 = fmul fast float %mul, %f1
  ret float %mul1
; CHECK-LABEL: @fmul6(
; CHECK: fmul fast float %f1, %f1
}

; "(X*Y) * X => (X*X) * Y" is disabled if "X*Y" has multiple uses
define float @fmul7(float %f1, float %f2) {
  %mul = fmul float %f1, %f2
  %mul1 = fmul fast float %mul, %f1
  %add = fadd float %mul1, %mul
  ret float %add
; CHECK-LABEL: @fmul7(
; CHECK: fmul fast float %mul, %f1
}

; =========================================================================
;
;   Testing-cases about negation
;
; =========================================================================
define float @fneg1(float %f1, float %f2) {
  %sub = fsub float -0.000000e+00, %f1
  %sub1 = fsub nsz float 0.000000e+00, %f2
  %mul = fmul float %sub, %sub1
  ret float %mul
; CHECK-LABEL: @fneg1(
; CHECK: fmul float %f1, %f2
}

define float @fneg2(float %x) {
  %sub = fsub nsz float 0.0, %x
  ret float %sub
; CHECK-LABEL: @fneg2(
; CHECK-NEXT: fsub nsz float -0.000000e+00, %x
; CHECK-NEXT: ret float 
}

; =========================================================================
;
;   Testing-cases about div
;
; =========================================================================

; X/C1 / C2 => X * (1/(C2*C1))
define float @fdiv1(float %x) {
  %div = fdiv float %x, 0x3FF3333340000000
  %div1 = fdiv fast float %div, 0x4002666660000000
  ret float %div1
; 0x3FF3333340000000 = 1.2f
; 0x4002666660000000 = 2.3f
; 0x3FD7303B60000000 = 0.36231884057971014492
; CHECK-LABEL: @fdiv1(
; CHECK: fmul fast float %x, 0x3FD7303B60000000
}

; X*C1 / C2 => X * (C1/C2)
define float @fdiv2(float %x) {
  %mul = fmul float %x, 0x3FF3333340000000
  %div1 = fdiv fast float %mul, 0x4002666660000000
  ret float %div1

; 0x3FF3333340000000 = 1.2f
; 0x4002666660000000 = 2.3f
; 0x3FE0B21660000000 = 0.52173918485641479492
; CHECK-LABEL: @fdiv2(
; CHECK: fmul fast float %x, 0x3FE0B21660000000
}

define <2 x float> @fdiv2_vec(<2 x float> %x) {
  %mul = fmul <2 x float> %x, <float 6.0, float 9.0>
  %div1 = fdiv fast <2 x float> %mul, <float 2.0, float 3.0>
  ret <2 x float> %div1

; CHECK-LABEL: @fdiv2_vec(
; CHECK: fmul fast <2 x float> %x, <float 3.000000e+00, float 3.000000e+00>
}

; "X/C1 / C2 => X * (1/(C2*C1))" is disabled (for now) is C2/C1 is a denormal
;
define float @fdiv3(float %x) {
  %div = fdiv float %x, 0x47EFFFFFE0000000
  %div1 = fdiv fast float %div, 0x4002666660000000
  ret float %div1
; CHECK-LABEL: @fdiv3(
; CHECK: fdiv float %x, 0x47EFFFFFE0000000
}

; "X*C1 / C2 => X * (C1/C2)" is disabled if C1/C2 is a denormal
define float @fdiv4(float %x) {
  %mul = fmul float %x, 0x47EFFFFFE0000000
  %div = fdiv float %mul, 0x3FC99999A0000000
  ret float %div
; CHECK-LABEL: @fdiv4(
; CHECK: fmul float %x, 0x47EFFFFFE0000000
}

; (X/Y)/Z = > X/(Y*Z)
define float @fdiv5(float %f1, float %f2, float %f3) {
  %t1 = fdiv float %f1, %f2
  %t2 = fdiv fast float %t1, %f3
  ret float %t2
; CHECK-LABEL: @fdiv5(
; CHECK: fmul float %f2, %f3
}

; Z/(X/Y) = > (Z*Y)/X
define float @fdiv6(float %f1, float %f2, float %f3) {
  %t1 = fdiv float %f1, %f2
  %t2 = fdiv fast float %f3, %t1
  ret float %t2
; CHECK-LABEL: @fdiv6(
; CHECK: fmul float %f3, %f2
}

; C1/(X*C2) => (C1/C2) / X
define float @fdiv7(float %x) {
  %t1 = fmul float %x, 3.0e0
  %t2 = fdiv fast float 15.0e0, %t1
  ret float %t2
; CHECK-LABEL: @fdiv7(
; CHECK: fdiv fast float 5.000000e+00, %x
}

; C1/(X/C2) => (C1*C2) / X
define float @fdiv8(float %x) {
  %t1 = fdiv float %x, 3.0e0
  %t2 = fdiv fast float 15.0e0, %t1
  ret float %t2
; CHECK-LABEL: @fdiv8(
; CHECK: fdiv fast float 4.500000e+01, %x
}

; C1/(C2/X) => (C1/C2) * X
define float @fdiv9(float %x) {
  %t1 = fdiv float 3.0e0, %x
  %t2 = fdiv fast float 15.0e0, %t1
  ret float %t2
; CHECK-LABEL: @fdiv9(
; CHECK: fmul fast float %x, 5.000000e+00
}

; =========================================================================
;
;   Testing-cases about factorization
;
; =========================================================================
; x*z + y*z => (x+y) * z
define float @fact_mul1(float %x, float %y, float %z) {
  %t1 = fmul fast float %x, %z
  %t2 = fmul fast float %y, %z
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fact_mul1(
; CHECK: fmul fast float %1, %z
}

; z*x + y*z => (x+y) * z
define float @fact_mul2(float %x, float %y, float %z) {
  %t1 = fmul fast float %z, %x
  %t2 = fmul fast float %y, %z
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fact_mul2(
; CHECK: fmul fast float %1, %z
}

; z*x - z*y => (x-y) * z
define float @fact_mul3(float %x, float %y, float %z) {
  %t2 = fmul fast float %z, %y
  %t1 = fmul fast float %z, %x
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fact_mul3(
; CHECK: fmul fast float %1, %z
}

; x*z - z*y => (x-y) * z
define float @fact_mul4(float %x, float %y, float %z) {
  %t1 = fmul fast float %x, %z
  %t2 = fmul fast float %z, %y
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fact_mul4(
; CHECK: fmul fast float %1, %z
}

; x/y + x/z, no xform
define float @fact_div1(float %x, float %y, float %z) {
  %t1 = fdiv fast float %x, %y
  %t2 = fdiv fast float %x, %z
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fact_div1
; CHECK: fadd fast float %t1, %t2
}

; x/y + z/x; no xform
define float @fact_div2(float %x, float %y, float %z) {
  %t1 = fdiv fast float %x, %y
  %t2 = fdiv fast float %z, %x
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fact_div2
; CHECK: fadd fast float %t1, %t2
}

; y/x + z/x => (y+z)/x
define float @fact_div3(float %x, float %y, float %z) {
  %t1 = fdiv fast float %y, %x
  %t2 = fdiv fast float %z, %x
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fact_div3
; CHECK: fdiv fast float %1, %x
}

; y/x - z/x => (y-z)/x
define float @fact_div4(float %x, float %y, float %z) {
  %t1 = fdiv fast float %y, %x
  %t2 = fdiv fast float %z, %x
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK: fact_div4
; CHECK: fdiv fast float %1, %x
}

; y/x - z/x => (y-z)/x is disabled if y-z is denormal.
define float @fact_div5(float %x) {
  %t1 = fdiv fast float 0x3810000000000000, %x
  %t2 = fdiv fast float 0x3800000000000000, %x
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fact_div5
; CHECK: fdiv fast float 0x3818000000000000, %x
}

; y/x - z/x => (y-z)/x is disabled if y-z is denormal.
define float @fact_div6(float %x) {
  %t1 = fdiv fast float 0x3810000000000000, %x
  %t2 = fdiv fast float 0x3800000000000000, %x
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK: fact_div6
; CHECK: %t3 = fsub fast float %t1, %t2
}

; =========================================================================
;
;   Test-cases for square root
;
; =========================================================================

; A squared factor fed into a square root intrinsic should be hoisted out
; as a fabs() value.

declare double @llvm.sqrt.f64(double)

define double @sqrt_intrinsic_arg_squared(double %x) {
  %mul = fmul fast double %x, %x
  %sqrt = call fast double @llvm.sqrt.f64(double %mul)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_arg_squared(
; CHECK-NEXT: %fabs = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: ret double %fabs
}

; Check all 6 combinations of a 3-way multiplication tree where
; one factor is repeated.

define double @sqrt_intrinsic_three_args1(double %x, double %y) {
  %mul = fmul fast double %y, %x
  %mul2 = fmul fast double %mul, %x
  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_three_args1(
; CHECK-NEXT: %fabs = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: %sqrt1 = call fast double @llvm.sqrt.f64(double %y)
; CHECK-NEXT: %1 = fmul fast double %fabs, %sqrt1
; CHECK-NEXT: ret double %1
}

define double @sqrt_intrinsic_three_args2(double %x, double %y) {
  %mul = fmul fast double %x, %y
  %mul2 = fmul fast double %mul, %x
  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_three_args2(
; CHECK-NEXT: %fabs = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: %sqrt1 = call fast double @llvm.sqrt.f64(double %y)
; CHECK-NEXT: %1 = fmul fast double %fabs, %sqrt1
; CHECK-NEXT: ret double %1
}

define double @sqrt_intrinsic_three_args3(double %x, double %y) {
  %mul = fmul fast double %x, %x
  %mul2 = fmul fast double %mul, %y
  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_three_args3(
; CHECK-NEXT: %fabs = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: %sqrt1 = call fast double @llvm.sqrt.f64(double %y)
; CHECK-NEXT: %1 = fmul fast double %fabs, %sqrt1
; CHECK-NEXT: ret double %1
}

define double @sqrt_intrinsic_three_args4(double %x, double %y) {
  %mul = fmul fast double %y, %x
  %mul2 = fmul fast double %x, %mul
  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_three_args4(
; CHECK-NEXT: %fabs = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: %sqrt1 = call fast double @llvm.sqrt.f64(double %y)
; CHECK-NEXT: %1 = fmul fast double %fabs, %sqrt1
; CHECK-NEXT: ret double %1
}

define double @sqrt_intrinsic_three_args5(double %x, double %y) {
  %mul = fmul fast double %x, %y
  %mul2 = fmul fast double %x, %mul
  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_three_args5(
; CHECK-NEXT: %fabs = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: %sqrt1 = call fast double @llvm.sqrt.f64(double %y)
; CHECK-NEXT: %1 = fmul fast double %fabs, %sqrt1
; CHECK-NEXT: ret double %1
}

define double @sqrt_intrinsic_three_args6(double %x, double %y) {
  %mul = fmul fast double %x, %x
  %mul2 = fmul fast double %y, %mul
  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_three_args6(
; CHECK-NEXT: %fabs = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: %sqrt1 = call fast double @llvm.sqrt.f64(double %y)
; CHECK-NEXT: %1 = fmul fast double %fabs, %sqrt1
; CHECK-NEXT: ret double %1
}

; If any operation is not 'fast', we can't simplify.

define double @sqrt_intrinsic_not_so_fast(double %x, double %y) {
  %mul = fmul double %x, %x
  %mul2 = fmul fast double %mul, %y
  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_not_so_fast(
; CHECK-NEXT:  %mul = fmul double %x, %x
; CHECK-NEXT:  %mul2 = fmul fast double %mul, %y
; CHECK-NEXT:  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
; CHECK-NEXT:  ret double %sqrt
}

define double @sqrt_intrinsic_arg_4th(double %x) {
  %mul = fmul fast double %x, %x
  %mul2 = fmul fast double %mul, %mul
  %sqrt = call fast double @llvm.sqrt.f64(double %mul2)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_arg_4th(
; CHECK-NEXT: %mul = fmul fast double %x, %x
; CHECK-NEXT: ret double %mul
}

define double @sqrt_intrinsic_arg_5th(double %x) {
  %mul = fmul fast double %x, %x
  %mul2 = fmul fast double %mul, %x
  %mul3 = fmul fast double %mul2, %mul
  %sqrt = call fast double @llvm.sqrt.f64(double %mul3)
  ret double %sqrt

; CHECK-LABEL: sqrt_intrinsic_arg_5th(
; CHECK-NEXT: %mul = fmul fast double %x, %x
; CHECK-NEXT: %sqrt1 = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT: %1 = fmul fast double %mul, %sqrt1
; CHECK-NEXT: ret double %1
}

; Check that square root calls have the same behavior.

declare float @sqrtf(float)
declare double @sqrt(double)
declare fp128 @sqrtl(fp128)

define float @sqrt_call_squared_f32(float %x) {
  %mul = fmul fast float %x, %x
  %sqrt = call fast float @sqrtf(float %mul)
  ret float %sqrt

; CHECK-LABEL: sqrt_call_squared_f32(
; CHECK-NEXT: %fabs = call fast float @llvm.fabs.f32(float %x)
; CHECK-NEXT: ret float %fabs
}

define double @sqrt_call_squared_f64(double %x) {
  %mul = fmul fast double %x, %x
  %sqrt = call fast double @sqrt(double %mul)
  ret double %sqrt

; CHECK-LABEL: sqrt_call_squared_f64(
; CHECK-NEXT: %fabs = call fast double @llvm.fabs.f64(double %x)
; CHECK-NEXT: ret double %fabs
}

define fp128 @sqrt_call_squared_f128(fp128 %x) {
  %mul = fmul fast fp128 %x, %x
  %sqrt = call fast fp128 @sqrtl(fp128 %mul)
  ret fp128 %sqrt

; CHECK-LABEL: sqrt_call_squared_f128(
; CHECK-NEXT: %fabs = call fast fp128 @llvm.fabs.f128(fp128 %x)
; CHECK-NEXT: ret fp128 %fabs
}

; =========================================================================
;
;   Test-cases for fmin / fmax
;
; =========================================================================

declare double @fmax(double, double)
declare double @fmin(double, double)
declare float @fmaxf(float, float)
declare float @fminf(float, float)
declare fp128 @fmaxl(fp128, fp128)
declare fp128 @fminl(fp128, fp128)

; No NaNs is the minimum requirement to replace these calls.
; This should always be set when unsafe-fp-math is true, but
; alternate the attributes for additional test coverage.
; 'nsz' is implied by the definition of fmax or fmin itself.

; Shrink and remove the call.
define float @max1(float %a, float %b) {
  %c = fpext float %a to double
  %d = fpext float %b to double
  %e = call fast double @fmax(double %c, double %d)
  %f = fptrunc double %e to float
  ret float %f

; CHECK-LABEL: max1(
; CHECK-NEXT:  fcmp fast ogt float %a, %b 
; CHECK-NEXT:  select {{.*}} float %a, float %b 
; CHECK-NEXT:  ret
}

define float @max2(float %a, float %b) {
  %c = call nnan float @fmaxf(float %a, float %b)
  ret float %c

; CHECK-LABEL: max2(
; CHECK-NEXT:  fcmp nnan nsz ogt float %a, %b 
; CHECK-NEXT:  select {{.*}} float %a, float %b 
; CHECK-NEXT:  ret
}


define double @max3(double %a, double %b) {
  %c = call fast double @fmax(double %a, double %b)
  ret double %c

; CHECK-LABEL: max3(
; CHECK-NEXT:  fcmp fast ogt double %a, %b 
; CHECK-NEXT:  select {{.*}} double %a, double %b 
; CHECK-NEXT:  ret
}

define fp128 @max4(fp128 %a, fp128 %b) {
  %c = call nnan fp128 @fmaxl(fp128 %a, fp128 %b)
  ret fp128 %c

; CHECK-LABEL: max4(
; CHECK-NEXT:  fcmp nnan nsz ogt fp128 %a, %b 
; CHECK-NEXT:  select {{.*}} fp128 %a, fp128 %b 
; CHECK-NEXT:  ret
}

; Shrink and remove the call.
define float @min1(float %a, float %b) {
  %c = fpext float %a to double
  %d = fpext float %b to double
  %e = call nnan double @fmin(double %c, double %d)
  %f = fptrunc double %e to float
  ret float %f

; CHECK-LABEL: min1(
; CHECK-NEXT:  fcmp nnan nsz olt float %a, %b 
; CHECK-NEXT:  select {{.*}} float %a, float %b 
; CHECK-NEXT:  ret
}

define float @min2(float %a, float %b) {
  %c = call fast float @fminf(float %a, float %b)
  ret float %c

; CHECK-LABEL: min2(
; CHECK-NEXT:  fcmp fast olt float %a, %b 
; CHECK-NEXT:  select {{.*}} float %a, float %b 
; CHECK-NEXT:  ret
}

define double @min3(double %a, double %b) {
  %c = call nnan double @fmin(double %a, double %b)
  ret double %c

; CHECK-LABEL: min3(
; CHECK-NEXT:  fcmp nnan nsz olt double %a, %b 
; CHECK-NEXT:  select {{.*}} double %a, double %b 
; CHECK-NEXT:  ret
}

define fp128 @min4(fp128 %a, fp128 %b) {
  %c = call fast fp128 @fminl(fp128 %a, fp128 %b)
  ret fp128 %c

; CHECK-LABEL: min4(
; CHECK-NEXT:  fcmp fast olt fp128 %a, %b 
; CHECK-NEXT:  select {{.*}} fp128 %a, fp128 %b 
; CHECK-NEXT:  ret
}
