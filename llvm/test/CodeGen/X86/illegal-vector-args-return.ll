; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=nehalem | grep "mulpd	%xmm3, %xmm1"
; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=nehalem | grep "mulpd	%xmm2, %xmm0"
; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=nehalem | grep "addps	%xmm3, %xmm1"
; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=nehalem | grep "addps	%xmm2, %xmm0"

target triple = "i686-apple-darwin8"

define <4 x double> @foo(<4 x double> %x, <4 x double> %z) {
  %y = fmul <4 x double> %x, %z
  ret <4 x double> %y
}

define <8 x float> @bar(<8 x float> %x, <8 x float> %z) {
  %y = fadd <8 x float> %x, %z
  ret <8 x float> %y
}
