; RUN: llc < %s -march=x86 -mcpu=yonah -mattr=+sse2,-sse41 | FileCheck %s

; CHECK: vsel_float
; CHECK: pandn
; CHECK: pand
; CHECK: por
; CHECK: ret
define void@vsel_float(<4 x float>* %v1, <4 x float>* %v2) {
  %A = load <4 x float>* %v1
  %B = load <4 x float>* %v2
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %A, <4 x float> %B
  store <4 x float > %vsel, <4 x float>* %v1
  ret void
}

; CHECK: vsel_i32
; CHECK: pandn
; CHECK: pand
; CHECK: por
; CHECK: ret
define void@vsel_i32(<4 x i32>* %v1, <4 x i32>* %v2) {
  %A = load <4 x i32>* %v1
  %B = load <4 x i32>* %v2
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i32> %A, <4 x i32> %B
  store <4 x i32 > %vsel, <4 x i32>* %v1
  ret void
}

; Without forcing instructions, fall back to the preferred PS domain.
; CHECK: vsel_i64
; CHECK: xorps
; CHECK: andps
; CHECK: andnps
; CHECK: orps
; CHECK: ret

define void@vsel_i64(<4 x i64>* %v1, <4 x i64>* %v2) {
  %A = load <4 x i64>* %v1
  %B = load <4 x i64>* %v2
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i64> %A, <4 x i64> %B
  store <4 x i64 > %vsel, <4 x i64>* %v1
  ret void
}

; Without forcing instructions, fall back to the preferred PS domain.
; CHECK: vsel_double
; CHECK: xorps
; CHECK: andps
; CHECK: andnps
; CHECK: orps
; CHECK: ret

define void@vsel_double(<4 x double>* %v1, <4 x double>* %v2) {
  %A = load <4 x double>* %v1
  %B = load <4 x double>* %v2
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x double> %A, <4 x double> %B
  store <4 x double > %vsel, <4 x double>* %v1
  ret void
}


