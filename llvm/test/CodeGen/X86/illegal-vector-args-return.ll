; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep {mulpd	%xmm3, %xmm1}
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep {mulpd	%xmm2, %xmm0}
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep {addps	%xmm3, %xmm1}
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep {addps	%xmm2, %xmm0}

define <4 x double> @foo(<4 x double> %x, <4 x double> %z) {
  %y = mul <4 x double> %x, %z
  ret <4 x double> %y
}

define <8 x float> @bar(<8 x float> %x, <8 x float> %z) {
  %y = add <8 x float> %x, %z
  ret <8 x float> %y
}
