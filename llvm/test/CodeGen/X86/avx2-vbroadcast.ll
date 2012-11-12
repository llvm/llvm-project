; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: vpbroadcastb (%
define <16 x i8> @BB16(i8* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i8* %ptr, align 4
  %q0 = insertelement <16 x i8> undef, i8 %q, i32 0
  %q1 = insertelement <16 x i8> %q0, i8 %q, i32 1
  %q2 = insertelement <16 x i8> %q1, i8 %q, i32 2
  %q3 = insertelement <16 x i8> %q2, i8 %q, i32 3
  %q4 = insertelement <16 x i8> %q3, i8 %q, i32 4
  %q5 = insertelement <16 x i8> %q4, i8 %q, i32 5
  %q6 = insertelement <16 x i8> %q5, i8 %q, i32 6
  %q7 = insertelement <16 x i8> %q6, i8 %q, i32 7
  %q8 = insertelement <16 x i8> %q7, i8 %q, i32 8
  %q9 = insertelement <16 x i8> %q8, i8 %q, i32 9
  %qa = insertelement <16 x i8> %q9, i8 %q, i32 10
  %qb = insertelement <16 x i8> %qa, i8 %q, i32 11
  %qc = insertelement <16 x i8> %qb, i8 %q, i32 12
  %qd = insertelement <16 x i8> %qc, i8 %q, i32 13
  %qe = insertelement <16 x i8> %qd, i8 %q, i32 14
  %qf = insertelement <16 x i8> %qe, i8 %q, i32 15
  ret <16 x i8> %qf
}
; CHECK: vpbroadcastb (%
define <32 x i8> @BB32(i8* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i8* %ptr, align 4
  %q0 = insertelement <32 x i8> undef, i8 %q, i32 0
  %q1 = insertelement <32 x i8> %q0, i8 %q, i32 1
  %q2 = insertelement <32 x i8> %q1, i8 %q, i32 2
  %q3 = insertelement <32 x i8> %q2, i8 %q, i32 3
  %q4 = insertelement <32 x i8> %q3, i8 %q, i32 4
  %q5 = insertelement <32 x i8> %q4, i8 %q, i32 5
  %q6 = insertelement <32 x i8> %q5, i8 %q, i32 6
  %q7 = insertelement <32 x i8> %q6, i8 %q, i32 7
  %q8 = insertelement <32 x i8> %q7, i8 %q, i32 8
  %q9 = insertelement <32 x i8> %q8, i8 %q, i32 9
  %qa = insertelement <32 x i8> %q9, i8 %q, i32 10
  %qb = insertelement <32 x i8> %qa, i8 %q, i32 11
  %qc = insertelement <32 x i8> %qb, i8 %q, i32 12
  %qd = insertelement <32 x i8> %qc, i8 %q, i32 13
  %qe = insertelement <32 x i8> %qd, i8 %q, i32 14
  %qf = insertelement <32 x i8> %qe, i8 %q, i32 15

  %q20 = insertelement <32 x i8> %qf, i8 %q,  i32 16
  %q21 = insertelement <32 x i8> %q20, i8 %q, i32 17
  %q22 = insertelement <32 x i8> %q21, i8 %q, i32 18
  %q23 = insertelement <32 x i8> %q22, i8 %q, i32 19
  %q24 = insertelement <32 x i8> %q23, i8 %q, i32 20
  %q25 = insertelement <32 x i8> %q24, i8 %q, i32 21
  %q26 = insertelement <32 x i8> %q25, i8 %q, i32 22
  %q27 = insertelement <32 x i8> %q26, i8 %q, i32 23
  %q28 = insertelement <32 x i8> %q27, i8 %q, i32 24
  %q29 = insertelement <32 x i8> %q28, i8 %q, i32 25
  %q2a = insertelement <32 x i8> %q29, i8 %q, i32 26
  %q2b = insertelement <32 x i8> %q2a, i8 %q, i32 27
  %q2c = insertelement <32 x i8> %q2b, i8 %q, i32 28
  %q2d = insertelement <32 x i8> %q2c, i8 %q, i32 29
  %q2e = insertelement <32 x i8> %q2d, i8 %q, i32 30
  %q2f = insertelement <32 x i8> %q2e, i8 %q, i32 31
  ret <32 x i8> %q2f
}
; CHECK: vpbroadcastw (%

define <8 x i16> @W16(i16* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i16* %ptr, align 4
  %q0 = insertelement <8 x i16> undef, i16 %q, i32 0
  %q1 = insertelement <8 x i16> %q0, i16 %q, i32 1
  %q2 = insertelement <8 x i16> %q1, i16 %q, i32 2
  %q3 = insertelement <8 x i16> %q2, i16 %q, i32 3
  %q4 = insertelement <8 x i16> %q3, i16 %q, i32 4
  %q5 = insertelement <8 x i16> %q4, i16 %q, i32 5
  %q6 = insertelement <8 x i16> %q5, i16 %q, i32 6
  %q7 = insertelement <8 x i16> %q6, i16 %q, i32 7
  ret <8 x i16> %q7
}
; CHECK: vpbroadcastw (%
define <16 x i16> @WW16(i16* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i16* %ptr, align 4
  %q0 = insertelement <16 x i16> undef, i16 %q, i32 0
  %q1 = insertelement <16 x i16> %q0, i16 %q, i32 1
  %q2 = insertelement <16 x i16> %q1, i16 %q, i32 2
  %q3 = insertelement <16 x i16> %q2, i16 %q, i32 3
  %q4 = insertelement <16 x i16> %q3, i16 %q, i32 4
  %q5 = insertelement <16 x i16> %q4, i16 %q, i32 5
  %q6 = insertelement <16 x i16> %q5, i16 %q, i32 6
  %q7 = insertelement <16 x i16> %q6, i16 %q, i32 7
  %q8 = insertelement <16 x i16> %q7, i16 %q, i32 8
  %q9 = insertelement <16 x i16> %q8, i16 %q, i32 9
  %qa = insertelement <16 x i16> %q9, i16 %q, i32 10
  %qb = insertelement <16 x i16> %qa, i16 %q, i32 11
  %qc = insertelement <16 x i16> %qb, i16 %q, i32 12
  %qd = insertelement <16 x i16> %qc, i16 %q, i32 13
  %qe = insertelement <16 x i16> %qd, i16 %q, i32 14
  %qf = insertelement <16 x i16> %qe, i16 %q, i32 15
  ret <16 x i16> %qf
}
; CHECK: vpbroadcastd (%
define <4 x i32> @D32(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32* %ptr, align 4
  %q0 = insertelement <4 x i32> undef, i32 %q, i32 0
  %q1 = insertelement <4 x i32> %q0, i32 %q, i32 1
  %q2 = insertelement <4 x i32> %q1, i32 %q, i32 2
  %q3 = insertelement <4 x i32> %q2, i32 %q, i32 3
  ret <4 x i32> %q3
}
; CHECK: vpbroadcastd (%
define <8 x i32> @DD32(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32* %ptr, align 4
  %q0 = insertelement <8 x i32> undef, i32 %q, i32 0
  %q1 = insertelement <8 x i32> %q0, i32 %q, i32 1
  %q2 = insertelement <8 x i32> %q1, i32 %q, i32 2
  %q3 = insertelement <8 x i32> %q2, i32 %q, i32 3
  %q4 = insertelement <8 x i32> %q3, i32 %q, i32 4
  %q5 = insertelement <8 x i32> %q4, i32 %q, i32 5
  %q6 = insertelement <8 x i32> %q5, i32 %q, i32 6
  %q7 = insertelement <8 x i32> %q6, i32 %q, i32 7
  ret <8 x i32> %q7
}
; CHECK: vpbroadcastq (%
define <2 x i64> @Q64(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64* %ptr, align 4
  %q0 = insertelement <2 x i64> undef, i64 %q, i32 0
  %q1 = insertelement <2 x i64> %q0, i64 %q, i32 1
  ret <2 x i64> %q1
}
; CHECK: vpbroadcastq (%
define <4 x i64> @QQ64(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64* %ptr, align 4
  %q0 = insertelement <4 x i64> undef, i64 %q, i32 0
  %q1 = insertelement <4 x i64> %q0, i64 %q, i32 1
  %q2 = insertelement <4 x i64> %q1, i64 %q, i32 2
  %q3 = insertelement <4 x i64> %q2, i64 %q, i32 3
  ret <4 x i64> %q3
}

; make sure that we still don't support broadcast double into 128-bit vector
; this used to crash
define <2 x double> @I(double* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load double* %ptr, align 4
  %vecinit.i = insertelement <2 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <2 x double> %vecinit.i, double %q, i32 1
  ret <2 x double> %vecinit2.i
}

; CHECK: V111
; CHECK: vpbroadcastd
; CHECK: ret
define <8 x i32> @V111(<8 x i32> %in) nounwind uwtable readnone ssp {
entry:
  %g = add <8 x i32> %in, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %g
}

; CHECK: V113
; CHECK: vbroadcastss
; CHECK: ret
define <8 x float> @V113(<8 x float> %in) nounwind uwtable readnone ssp {
entry:
  %g = fadd <8 x float> %in, <float 0xbf80000000000000, float 0xbf80000000000000, float 0xbf80000000000000, float 0xbf80000000000000, float 0xbf80000000000000, float 0xbf80000000000000, float 0xbf80000000000000, float 0xbf80000000000000>
  ret <8 x float> %g
}

; CHECK: _e2
; CHECK: vbroadcastss
; CHECK: ret
define <4 x float> @_e2(float* %ptr) nounwind uwtable readnone ssp {
  %vecinit.i = insertelement <4 x float> undef, float        0xbf80000000000000, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float  0xbf80000000000000, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float 0xbf80000000000000, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float 0xbf80000000000000, i32 3
  ret <4 x float> %vecinit6.i
}

; CHECK: _e4
; CHECK-NOT: broadcast
; CHECK: ret
define <8 x i8> @_e4(i8* %ptr) nounwind uwtable readnone ssp {
  %vecinit0.i = insertelement <8 x i8> undef, i8       52, i32 0
  %vecinit1.i = insertelement <8 x i8> %vecinit0.i, i8 52, i32 1
  %vecinit2.i = insertelement <8 x i8> %vecinit1.i, i8 52, i32 2
  %vecinit3.i = insertelement <8 x i8> %vecinit2.i, i8 52, i32 3
  %vecinit4.i = insertelement <8 x i8> %vecinit3.i, i8 52, i32 4
  %vecinit5.i = insertelement <8 x i8> %vecinit4.i, i8 52, i32 5
  %vecinit6.i = insertelement <8 x i8> %vecinit5.i, i8 52, i32 6
  %vecinit7.i = insertelement <8 x i8> %vecinit6.i, i8 52, i32 7
  ret <8 x i8> %vecinit7.i
}


define void @crash() nounwind alwaysinline {
WGLoopsEntry:
  br i1 undef, label %ret, label %footer329VF

footer329VF:
  %A.0.inVF = fmul float undef, 6.553600e+04
  %B.0.in407VF = fmul <8 x float> undef, <float 6.553600e+04, float 6.553600e+04, float 6.553600e+04, float 6.553600e+04, float 6.553600e+04, float 6.553600e+04, float 6.553600e+04, float 6.553600e+04>
  %A.0VF = fptosi float %A.0.inVF to i32
  %B.0408VF = fptosi <8 x float> %B.0.in407VF to <8 x i32>
  %0 = and <8 x i32> %B.0408VF, <i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535, i32 65535>
  %1 = and i32 %A.0VF, 65535
  %temp1098VF = insertelement <8 x i32> undef, i32 %1, i32 0
  %vector1099VF = shufflevector <8 x i32> %temp1098VF, <8 x i32> undef, <8 x i32> zeroinitializer
  br i1 undef, label %preload1201VF, label %footer349VF

preload1201VF:
  br label %footer349VF

footer349VF:
  %2 = mul nsw <8 x i32> undef, %0
  %3 = mul nsw <8 x i32> undef, %vector1099VF
  br label %footer329VF

ret:
  ret void
}

; CHECK: _inreg0
; CHECK: broadcastss
; CHECK: ret
define <8 x i32> @_inreg0(i32 %scalar) nounwind uwtable readnone ssp {
  %in = insertelement <8 x i32> undef, i32 %scalar, i32 0
  %wide = shufflevector <8 x i32> %in, <8 x i32> undef, <8 x i32> zeroinitializer
  ret <8 x i32> %wide
}

; CHECK: _inreg1
; CHECK: broadcastss
; CHECK: ret
define <8 x float> @_inreg1(float %scalar) nounwind uwtable readnone ssp {
  %in = insertelement <8 x float> undef, float %scalar, i32 0
  %wide = shufflevector <8 x float> %in, <8 x float> undef, <8 x i32> zeroinitializer
  ret <8 x float> %wide
}

; CHECK: _inreg2
; CHECK: broadcastss
; CHECK: ret
define <4 x float> @_inreg2(float %scalar) nounwind uwtable readnone ssp {
  %in = insertelement <4 x float> undef, float %scalar, i32 0
  %wide = shufflevector <4 x float> %in, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %wide
}

; CHECK: _inreg3
; CHECK: broadcastsd
; CHECK: ret
define <4 x double> @_inreg3(double %scalar) nounwind uwtable readnone ssp {
  %in = insertelement <4 x double> undef, double %scalar, i32 0
  %wide = shufflevector <4 x double> %in, <4 x double> undef, <4 x i32> zeroinitializer
  ret <4 x double> %wide
}

;CHECK: _inreg8xfloat
;CHECK: vbroadcastss
;CHECK: ret
define   <8 x float> @_inreg8xfloat(<8 x float> %a) {
  %b = shufflevector <8 x float> %a, <8 x float> undef, <8 x i32> zeroinitializer
  ret <8 x float> %b
}

;CHECK: _inreg4xfloat
;CHECK: vbroadcastss
;CHECK: ret
define   <4 x float> @_inreg4xfloat(<4 x float> %a) {
  %b = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %b
}

;CHECK: _inreg16xi16
;CHECK: vpbroadcastw
;CHECK: ret
define   <16 x i16> @_inreg16xi16(<16 x i16> %a) {
  %b = shufflevector <16 x i16> %a, <16 x i16> undef, <16 x i32> zeroinitializer
  ret <16 x i16> %b
}

;CHECK: _inreg8xi16
;CHECK: vpbroadcastw
;CHECK: ret
define   <8 x i16> @_inreg8xi16(<8 x i16> %a) {
  %b = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %b
}


;CHECK: _inreg4xi64
;CHECK: vpbroadcastq
;CHECK: ret
define   <4 x i64> @_inreg4xi64(<4 x i64> %a) {
  %b = shufflevector <4 x i64> %a, <4 x i64> undef, <4 x i32> zeroinitializer
  ret <4 x i64> %b
}

;CHECK: _inreg2xi64
;CHECK: vpbroadcastq
;CHECK: ret
define   <2 x i64> @_inreg2xi64(<2 x i64> %a) {
  %b = shufflevector <2 x i64> %a, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %b
}

;CHECK: _inreg4xdouble
;CHECK: vbroadcastsd
;CHECK: ret
define   <4 x double> @_inreg4xdouble(<4 x double> %a) {
  %b = shufflevector <4 x double> %a, <4 x double> undef, <4 x i32> zeroinitializer
  ret <4 x double> %b
}

;CHECK: _inreg2xdouble
;CHECK: vpbroadcastq
;CHECK: ret
define   <2 x double> @_inreg2xdouble(<2 x double> %a) {
  %b = shufflevector <2 x double> %a, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %b
}

;CHECK: _inreg8xi32
;CHECK: vpbroadcastd
;CHECK: ret
define   <8 x i32> @_inreg8xi32(<8 x i32> %a) {
  %b = shufflevector <8 x i32> %a, <8 x i32> undef, <8 x i32> zeroinitializer
  ret <8 x i32> %b
}

;CHECK: _inreg4xi32
;CHECK: vpbroadcastd
;CHECK: ret
define   <4 x i32> @_inreg4xi32(<4 x i32> %a) {
  %b = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %b
}

;CHECK: _inreg32xi8
;CHECK: vpbroadcastb
;CHECK: ret
define   <32 x i8> @_inreg32xi8(<32 x i8> %a) {
  %b = shufflevector <32 x i8> %a, <32 x i8> undef, <32 x i32> zeroinitializer
  ret <32 x i8> %b
}

;CHECK: _inreg16xi8
;CHECK: vpbroadcastb
;CHECK: ret
define   <16 x i8> @_inreg16xi8(<16 x i8> %a) {
  %b = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %b
}
