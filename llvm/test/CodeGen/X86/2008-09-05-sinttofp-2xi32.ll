; RUN: llc < %s -march=x86 -mattr=+sse2,+mmx | FileCheck %s
; originally from PR2687, but things don't work that way any more.
; there are no MMX instructions here; we use XMM.

define <2 x double> @a(<2 x i32> %x) nounwind {
entry:
; CHECK-LABEL: a
; CHECK-NOT: unpcklpd
  %y = sitofp <2 x i32> %x to <2 x double>
  ret <2 x double> %y
}

define <2 x i32> @b(<2 x double> %x) nounwind {
entry:
; CHECK-LABEL: b
; CHECK-NOT: unpckhpd
  %y = fptosi <2 x double> %x to <2 x i32>
  ret <2 x i32> %y
}

; This is how to get MMX instructions.

define <2 x double> @a2(x86_mmx %x) nounwind {
entry:
; CHECK-LABEL: a2
; CHECK: cvtpi2pd
; CHECK-NOT: cvtpi2pd
  %y = tail call <2 x double> @llvm.x86.sse.cvtpi2pd(x86_mmx %x)
  ret <2 x double> %y
}

define x86_mmx @b2(<2 x double> %x) nounwind {
entry:
; CHECK-LABEL: b2
; CHECK: cvttpd2pi
; CHECK-NOT: cvttpd2pi
  %y = tail call x86_mmx @llvm.x86.sse.cvttpd2pi (<2 x double> %x)
  ret x86_mmx %y
}

declare <2 x double> @llvm.x86.sse.cvtpi2pd(x86_mmx)
declare x86_mmx @llvm.x86.sse.cvttpd2pi(<2 x double>)
