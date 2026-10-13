; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 -denormal-fp-math-f32=preserve-sign | FileCheck %s
; RUN: %if ptxas-sm_80 && ptxas-isa-7.0 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 -denormal-fp-math-f32=preserve-sign | %ptxas-verify -arch=sm_80 %}

; Check for working ftz on f16 to be sure
define half @neg_f16_ftz(half %a) {
; CHECK-LABEL: neg_f16_ftz
; CHECK: neg.ftz.f16
  %r = fneg half %a
  ret half %r
}

define bfloat @neg_bf16_no_ftz(bfloat %a) {
; CHECK-LABEL: neg_bf16_no_ftz
; CHECK: neg.bf16
  %r = fneg bfloat %a
  ret bfloat %r
}

define <2 x bfloat> @neg_bf16x2_no_ftz(<2 x bfloat> %a) {
; CHECK-LABEL: neg_bf16x2_no_ftz
; CHECK: neg.bf16x2
  %r = fneg <2 x bfloat> %a
  ret <2 x bfloat> %r
}
