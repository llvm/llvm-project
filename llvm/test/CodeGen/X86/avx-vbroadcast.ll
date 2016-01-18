; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vbroadcastsd (%
define <4 x i64> @A(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64, i64* %ptr, align 8
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %vecinit2.i = insertelement <4 x i64> %vecinit.i, i64 %q, i32 1
  %vecinit4.i = insertelement <4 x i64> %vecinit2.i, i64 %q, i32 2
  %vecinit6.i = insertelement <4 x i64> %vecinit4.i, i64 %q, i32 3
  ret <4 x i64> %vecinit6.i
}

; CHECK: vbroadcastss (%
define <8 x i32> @B(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32, i32* %ptr, align 4
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %vecinit2.i = insertelement <8 x i32> %vecinit.i, i32 %q, i32 1
  %vecinit4.i = insertelement <8 x i32> %vecinit2.i, i32 %q, i32 2
  %vecinit6.i = insertelement <8 x i32> %vecinit4.i, i32 %q, i32 3
  ret <8 x i32> %vecinit6.i
}

; CHECK: vbroadcastsd (%
define <4 x double> @C(double* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load double, double* %ptr, align 8
  %vecinit.i = insertelement <4 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <4 x double> %vecinit.i, double %q, i32 1
  %vecinit4.i = insertelement <4 x double> %vecinit2.i, double %q, i32 2
  %vecinit6.i = insertelement <4 x double> %vecinit4.i, double %q, i32 3
  ret <4 x double> %vecinit6.i
}

; CHECK: vbroadcastss (%
define <8 x float> @D(float* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load float, float* %ptr, align 4
  %vecinit.i = insertelement <8 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <8 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <8 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <8 x float> %vecinit4.i, float %q, i32 3
  ret <8 x float> %vecinit6.i
}

;;;; 128-bit versions

; CHECK: vbroadcastss (%
define <4 x float> @e(float* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load float, float* %ptr, align 4
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  ret <4 x float> %vecinit6.i
}


; CHECK: _e2
; CHECK-NOT: vbroadcastss
; CHECK: ret
define <4 x float> @_e2(float* %ptr) nounwind uwtable readnone ssp {
    %vecinit.i = insertelement <4 x float> undef, float      0xbf80000000000000, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float  0xbf80000000000000, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float 0xbf80000000000000, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float 0xbf80000000000000, i32 3
  ret <4 x float> %vecinit6.i
}


; CHECK: vbroadcastss (%
define <4 x i32> @F(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32, i32* %ptr, align 4
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %vecinit2.i = insertelement <4 x i32> %vecinit.i, i32 %q, i32 1
  %vecinit4.i = insertelement <4 x i32> %vecinit2.i, i32 %q, i32 2
  %vecinit6.i = insertelement <4 x i32> %vecinit4.i, i32 %q, i32 3
  ret <4 x i32> %vecinit6.i
}

; Unsupported vbroadcasts

; CHECK: _G
; CHECK-NOT: broadcast (%
; CHECK: ret
define <2 x i64> @G(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64, i64* %ptr, align 8
  %vecinit.i = insertelement <2 x i64> undef, i64 %q, i32 0
  %vecinit2.i = insertelement <2 x i64> %vecinit.i, i64 %q, i32 1
  ret <2 x i64> %vecinit2.i
}

; CHECK: _H
; CHECK-NOT: broadcast
; CHECK: ret
define <4 x i32> @H(<4 x i32> %a) {
  %x = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  ret <4 x i32> %x
}

; CHECK: _I
; CHECK-NOT: broadcast (%
; CHECK: ret
define <2 x double> @I(double* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load double, double* %ptr, align 4
  %vecinit.i = insertelement <2 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <2 x double> %vecinit.i, double %q, i32 1
  ret <2 x double> %vecinit2.i
}

; CHECK: _RR
; CHECK: vbroadcastss (%
; CHECK: ret
define <4 x float> @_RR(float* %ptr, i32* %k) nounwind uwtable readnone ssp {
entry:
  %q = load float, float* %ptr, align 4
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  ; force a chain
  %j = load i32, i32* %k, align 4
  store i32 %j, i32* undef
  ret <4 x float> %vecinit6.i
}


; CHECK: _RR2
; CHECK: vbroadcastss (%
; CHECK: ret
define <4 x float> @_RR2(float* %ptr, i32* %k) nounwind uwtable readnone ssp {
entry:
  %q = load float, float* %ptr, align 4
  %v = insertelement <4 x float> undef, float %q, i32 0
  %t = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %t
}


; These tests check that a vbroadcast instruction is used when we have a splat
; formed from a concat_vectors (via the shufflevector) of two BUILD_VECTORs
; (via the insertelements).

; CHECK-LABEL: splat_concat1
; CHECK-NOT: vinsertf128
; CHECK: vbroadcastss (%
; CHECK-NEXT: ret
define <8 x float> @splat_concat1(float* %p) {
  %1 = load float, float* %p, align 4
  %2 = insertelement <4 x float> undef, float %1, i32 0
  %3 = insertelement <4 x float> %2, float %1, i32 1
  %4 = insertelement <4 x float> %3, float %1, i32 2
  %5 = insertelement <4 x float> %4, float %1, i32 3
  %6 = shufflevector <4 x float> %5, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  ret <8 x float> %6
}

; CHECK-LABEL: splat_concat2
; CHECK-NOT: vinsertf128
; CHECK: vbroadcastss (%
; CHECK-NEXT: ret
define <8 x float> @splat_concat2(float* %p) {
  %1 = load float, float* %p, align 4
  %2 = insertelement <4 x float> undef, float %1, i32 0
  %3 = insertelement <4 x float> %2, float %1, i32 1
  %4 = insertelement <4 x float> %3, float %1, i32 2
  %5 = insertelement <4 x float> %4, float %1, i32 3
  %6 = insertelement <4 x float> undef, float %1, i32 0
  %7 = insertelement <4 x float> %6, float %1, i32 1
  %8 = insertelement <4 x float> %7, float %1, i32 2
  %9 = insertelement <4 x float> %8, float %1, i32 3
  %10 = shufflevector <4 x float> %5, <4 x float> %9, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %10
}

; CHECK-LABEL: splat_concat3
; CHECK-NOT: vinsertf128
; CHECK: vbroadcastsd (%
; CHECK-NEXT: ret
define <4 x double> @splat_concat3(double* %p) {
  %1 = load double, double* %p, align 8
  %2 = insertelement <2 x double> undef, double %1, i32 0
  %3 = insertelement <2 x double> %2, double %1, i32 1
  %4 = shufflevector <2 x double> %3, <2 x double> undef, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  ret <4 x double> %4
}

; CHECK-LABEL: splat_concat4
; CHECK-NOT: vinsertf128
; CHECK: vbroadcastsd (%
; CHECK-NEXT: ret
define <4 x double> @splat_concat4(double* %p) {
  %1 = load double, double* %p, align 8
  %2 = insertelement <2 x double> undef, double %1, i32 0
  %3 = insertelement <2 x double> %2, double %1, i32 1
  %4 = insertelement <2 x double> undef, double %1, i32 0
  %5 = insertelement <2 x double> %2, double %1, i32 1
  %6 = shufflevector <2 x double> %3, <2 x double> %5, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x double> %6
}

