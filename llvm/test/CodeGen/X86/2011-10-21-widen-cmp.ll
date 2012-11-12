; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; Check that a <4 x float> compare is generated and that we are
; not stuck in an endless loop.

; CHECK: cmp_2_floats
; CHECK: cmpordps
; CHECK: ret

define void @cmp_2_floats() {
entry:
  %0 = fcmp oeq <2 x float> undef, undef
  %1 = select <2 x i1> %0, <2 x float> undef, <2 x float> undef
  store <2 x float> %1, <2 x float>* undef
  ret void
}

; CHECK: cmp_2_doubles
; CHECK: cmpordpd
; CHECK: blendvpd
; CHECK: ret
define void @cmp_2_doubles() {
entry:
  %0 = fcmp oeq <2 x double> undef, undef
  %1 = select <2 x i1> %0, <2 x double> undef, <2 x double> undef
  store <2 x double> %1, <2 x double>* undef
  ret void
}

; CHECK: mp_11193
; CHECK: psraw   $15
; CHECK: ret
define void @mp_11193(<8 x float> * nocapture %aFOO, <8 x float>* nocapture %RET)
nounwind {
allocas:
  %bincmp = fcmp olt <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 9.000000e+00, float 1.000000e+00, float 9.000000e+00, float 1.000000e+00> , <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  %t = extractelement <8 x i1> %bincmp, i32 0
  %ft = sitofp i1 %t to float
  %pp = bitcast <8 x float>* %RET to float*
  store float %ft, float* %pp
  ret void
}

