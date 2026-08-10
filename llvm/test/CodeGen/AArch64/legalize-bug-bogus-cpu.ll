; RUN: llc < %s -mtriple=aarch64 -mcpu=bogus

; Fix the bug in PR20557. Set mcpu to a bogus name, llc will crash in type
; legalization.
define <4 x float> @fneg4(<4 x float> %x) {
  %sub = fsub <4 x float> zeroinitializer, %x
  ret <4 x float> %sub
}
