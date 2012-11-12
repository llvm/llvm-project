; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vbroadcastsd (%
define <4 x i64> @A(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64* %ptr, align 8
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %vecinit2.i = insertelement <4 x i64> %vecinit.i, i64 %q, i32 1
  %vecinit4.i = insertelement <4 x i64> %vecinit2.i, i64 %q, i32 2
  %vecinit6.i = insertelement <4 x i64> %vecinit4.i, i64 %q, i32 3
  ret <4 x i64> %vecinit6.i
}

; CHECK: vbroadcastss (%
define <8 x i32> @B(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32* %ptr, align 4
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %vecinit2.i = insertelement <8 x i32> %vecinit.i, i32 %q, i32 1
  %vecinit4.i = insertelement <8 x i32> %vecinit2.i, i32 %q, i32 2
  %vecinit6.i = insertelement <8 x i32> %vecinit4.i, i32 %q, i32 3
  ret <8 x i32> %vecinit6.i
}

; CHECK: vbroadcastsd (%
define <4 x double> @C(double* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load double* %ptr, align 8
  %vecinit.i = insertelement <4 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <4 x double> %vecinit.i, double %q, i32 1
  %vecinit4.i = insertelement <4 x double> %vecinit2.i, double %q, i32 2
  %vecinit6.i = insertelement <4 x double> %vecinit4.i, double %q, i32 3
  ret <4 x double> %vecinit6.i
}

; CHECK: vbroadcastss (%
define <8 x float> @D(float* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load float* %ptr, align 4
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
  %q = load float* %ptr, align 4
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
  %q = load i32* %ptr, align 4
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
  %q = load i64* %ptr, align 8
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
  %q = load double* %ptr, align 4
  %vecinit.i = insertelement <2 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <2 x double> %vecinit.i, double %q, i32 1
  ret <2 x double> %vecinit2.i
}

; CHECK: _RR
; CHECK: vbroadcastss (%
; CHECK: ret
define <4 x float> @_RR(float* %ptr, i32* %k) nounwind uwtable readnone ssp {
entry:
  %q = load float* %ptr, align 4
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  ; force a chain
  %j = load i32* %k, align 4
  store i32 %j, i32* undef
  ret <4 x float> %vecinit6.i
}


; CHECK: _RR2
; CHECK: vbroadcastss (%
; CHECK: ret
define <4 x float> @_RR2(float* %ptr, i32* %k) nounwind uwtable readnone ssp {
entry:
  %q = load float* %ptr, align 4
  %v = insertelement <4 x float> undef, float %q, i32 0
  %t = shufflevector <4 x float> %v, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %t
}

