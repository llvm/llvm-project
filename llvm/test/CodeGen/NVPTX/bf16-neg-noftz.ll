; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 -denormal-fp-math-f32=preserve-sign | FileCheck %s
; RUN: %if ptxas-sm_80 && ptxas-isa-7.0 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 -denormal-fp-math-f32=preserve-sign | %ptxas-verify -arch=sm_80 %}

; fneg whose result is observed as raw bits (returned) expands to a sign-bit
; XOR to preserve NaN payloads rather than using native neg.
define half @neg_f16_ftz(half %a) {
; CHECK-LABEL: neg_f16_ftz
; CHECK: xor.b16 {{.*}}, -32768
  %r = fneg half %a
  ret half %r
}

define bfloat @neg_bf16_no_ftz(bfloat %a) {
; CHECK-LABEL: neg_bf16_no_ftz
; CHECK: xor.b16 {{.*}}, -32768
  %r = fneg bfloat %a
  ret bfloat %r
}

define <2 x bfloat> @neg_bf16x2_no_ftz(<2 x bfloat> %a) {
; CHECK-LABEL: neg_bf16x2_no_ftz
; CHECK: xor.b32 {{.*}}, -2147450880
  %r = fneg <2 x bfloat> %a
  ret <2 x bfloat> %r
}

; Check for working ftz on f16 to be sure
define half @neg_fma_f16_ftz(half %a, half %b, half %c) {
; CHECK-LABEL: neg_fma_f16_ftz
; CHECK: neg.ftz.f16
  %n = fneg half %a
  %r = call half @llvm.fma.f16(half %n, half %b, half %c)
  ret half %r
}

define bfloat @neg_fma_bf16_no_ftz(bfloat %a, bfloat %b, bfloat %c) {
; CHECK-LABEL: neg_fma_bf16_no_ftz
; CHECK: neg.bf16
  %n = fneg bfloat %a
  %r = call bfloat @llvm.fma.bf16(bfloat %n, bfloat %b, bfloat %c)
  ret bfloat %r
}
