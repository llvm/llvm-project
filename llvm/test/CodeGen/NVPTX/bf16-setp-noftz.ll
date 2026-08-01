; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 -denormal-fp-math-f32=preserve-sign | FileCheck %s
; RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 -denormal-fp-math-f32=preserve-sign | %ptxas-verify -arch=sm_90 %}

; Check for working ftz on f16 to be sure
define i1 @setp_f16_ftz(half %a, half %b) {
; CHECK-LABEL: setp_f16_ftz
; CHECK: setp.neu.ftz.f16
  %cmp = fcmp une half %a, %b
  ret i1 %cmp
}

define i1 @setp_bf16_no_ftz(bfloat %a, bfloat %b) {
; CHECK-LABEL: setp_bf16_no_ftz
; CHECK: setp.neu.bf16
  %cmp = fcmp une bfloat %a, %b
  ret i1 %cmp
}

define <2 x bfloat> @setp_bf16x2_no_ftz(<2 x bfloat> %a, <2 x bfloat> %b,
  <2 x bfloat> %x, <2 x bfloat> %y) {
; CHECK-LABEL: setp_bf16x2_no_ftz
; CHECK: setp.neu.bf16x2
  %cmp = fcmp une <2 x bfloat> %a, %b
  %sel = select <2 x i1> %cmp, <2 x bfloat> %x, <2 x bfloat> %y
  ret <2 x bfloat> %sel
}
