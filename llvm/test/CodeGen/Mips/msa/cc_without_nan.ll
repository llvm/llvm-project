; RUN: llc -mtriple mips64-unknown-linux -mcpu=mips64r5 -mattr=+msa < %s | FileCheck %s

; The fcmp fast flag will result in conversion from
; setolt, setoeq, setole, setone to
; setlt, seteq, setle, setne nodes.
; Test that the latter nodes are matched to the same instructions as the former.

define <2 x i1> @testlt_v2f64(<2 x double> %a, <2 x double> %b) {
start:
  %0 = fcmp fast olt <2 x double> %a, %b
  ; CHECK: fclt.d
  ret <2 x i1> %0
}

define <4 x i1>  @testlt_v4f32(<4 x float> %a, <4 x float> %b) {
start:
  %0 = fcmp fast olt <4 x float> %a, %b
  ; CHECK: fclt.w
  ret <4 x i1> %0
}

define <2 x i1> @testeq_v2f64(<2 x double> %a, <2 x double> %b) {
start:
  %0 = fcmp fast oeq <2 x double> %a, %b
  ; CHECK: fceq.d
  ret <2 x i1> %0
}

define <4 x i1> @testeq_v4f32(<4 x float> %a, <4 x float> %b) {
start:
  %0 = fcmp fast oeq <4 x float> %a, %b
  ; CHECK: fceq.w
  ret <4 x i1> %0
}

define <2 x i1> @testle_v2f64(<2 x double> %a, <2 x double> %b) {
start:
  %0 = fcmp fast ole <2 x double> %a, %b
  ; CHECK: fcle.d
  ret <2 x i1> %0
}

define <4 x i1> @testle_v4f32(<4 x float> %a, <4 x float> %b) {
start:
  %0 = fcmp fast ole <4 x float> %a, %b
  ; CHECK: fcle.w
  ret <4 x i1> %0
}

define <2 x i1> @testne_v2f64(<2 x double> %a, <2 x double> %b) {
start:
  %0 = fcmp fast one <2 x double> %a, %b
  ; CHECK: fcne.d
  ret <2 x i1> %0
}

define <4 x i1> @testne_v4f32(<4 x float> %a, <4 x float> %b) {
start:
  %0 = fcmp fast one <4 x float> %a, %b
  ; CHECK: fcne.w
  ret <4 x i1> %0
}

