; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 -mattr=+sse41 | FileCheck %s

;CHECK: vsel_float
;CHECK: blendvps
;CHECK: ret
define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}


;CHECK: vsel_4xi8
;CHECK: blendvps
;CHECK: ret
define <4 x i8> @vsel_4xi8(<4 x i8> %v1, <4 x i8> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i8> %v1, <4 x i8> %v2
  ret <4 x i8> %vsel
}

;CHECK: vsel_4xi16
;CHECK: blendvps
;CHECK: ret
define <4 x i16> @vsel_4xi16(<4 x i16> %v1, <4 x i16> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i16> %v1, <4 x i16> %v2
  ret <4 x i16> %vsel
}


;CHECK: vsel_i32
;CHECK: blendvps
;CHECK: ret
define <4 x i32> @vsel_i32(<4 x i32> %v1, <4 x i32> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i32> %v1, <4 x i32> %v2
  ret <4 x i32> %vsel
}


;CHECK: vsel_double
;CHECK: blendvpd
;CHECK: ret
define <4 x double> @vsel_double(<4 x double> %v1, <4 x double> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x double> %v1, <4 x double> %v2
  ret <4 x double> %vsel
}


;CHECK: vsel_i64
;CHECK: blendvpd
;CHECK: ret
define <4 x i64> @vsel_i64(<4 x i64> %v1, <4 x i64> %v2) {
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i64> %v1, <4 x i64> %v2
  ret <4 x i64> %vsel
}


;CHECK: vsel_i8
;CHECK: pblendvb
;CHECK: ret
define <16 x i8> @vsel_i8(<16 x i8> %v1, <16 x i8> %v2) {
  %vsel = select <16 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <16 x i8> %v1, <16 x i8> %v2
  ret <16 x i8> %vsel
}

;; TEST blend + compares
; CHECK: A
define <2 x double> @A(<2 x double> %x, <2 x double> %y) {
  ; CHECK: cmplepd
  ; CHECK: blendvpd
  %max_is_x = fcmp oge <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

; CHECK: B
define <2 x double> @B(<2 x double> %x, <2 x double> %y) {
  ; CHECK: cmpnlepd
  ; CHECK: blendvpd
  %min_is_x = fcmp ult <2 x double> %x, %y
  %min = select <2 x i1> %min_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %min
}

; CHECK: float_crash
define void @float_crash() nounwind {
entry:
  %merge205vector_func.i = select <4 x i1> undef, <4 x double> undef, <4 x double> undef
  %extract214vector_func.i = extractelement <4 x double> %merge205vector_func.i, i32 0
  store double %extract214vector_func.i, double addrspace(1)* undef, align 8
  ret void
}
