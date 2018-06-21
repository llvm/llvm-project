; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown-unknown -mcpu=generic -mattr=+avx < %s | FileCheck %s

; When commuting a VADDSDrr instruction, verify that the 'IsUndef' flag is
; correctly propagated to the operands of the resulting instruction.
; Test for PR23103;

declare zeroext i1 @foo(<1 x double>)

define <1 x double> @pr23103(<1 x double>* align 8 %Vp) {
; CHECK-LABEL: pr23103:
; CHECK:         vmovsd (%rdi), %xmm0
; CHECK-NEXT:    vmovsd %xmm0, {{.*}}(%rsp) {{.*#+}} 8-byte Spill
; CHECK-NEXT:    callq foo
; CHECK-NEXT:    vaddsd {{.*}}(%rsp), %xmm0, %xmm0 {{.*#+}} 8-byte Folded Reload
; CHECK:         retq
entry:
  %V = load <1 x double>, <1 x double>* %Vp, align 8
  %call = call zeroext i1 @foo(<1 x double> %V)
  %fadd = fadd <1 x double> %V, undef
  ret <1 x double> %fadd
}
