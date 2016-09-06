; RUN: llc -mtriple=aarch64 < %s | FileCheck %s

define <8 x i8> @float_to_i8(<8 x float>* %in) {
; CHECK-LABEL: float_to_i8:
; CHECK: ldp     q1, q0, [x0]
; CHECK-DAG: fadd v[[LSB:[0-9]+]].4s, v1.4s, v1.4s
; CHECK-DAG: fadd v[[MSB:[0-9]+]].4s, v0.4s, v0.4s
; CHECK-DAG: fcvtzu v[[LSB2:[0-9]+]].4s, v[[LSB]].4s
; CHECK-DAG: fcvtzu v[[MSB2:[0-9]+]].4s, v[[MSB]].4s
; CHECK-DAG: xtn v[[TMP:[0-9]+]].4h, v[[LSB]].4s
; CHECK-DAG: xtn2 v[[TMP]].8h, v[[MSB]].4s
; CHECK-DAG: xtn v0.8b, v[[TMP]].8h
  %l = load <8 x float>, <8 x float>* %in
  %scale = fmul <8 x float> %l, <float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0, float 2.0>
  %conv = fptoui <8 x float> %scale to <8 x i8>
  ret <8 x i8> %conv
}
