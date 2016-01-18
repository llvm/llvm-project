; RUN: llc %s -mtriple=x86_64-unknown-unknown -mattr='-sse4.1' -o - | FileCheck %s -check-prefix=NO_SSE_41
; RUN: llc %s -mtriple=x86_64-unknown-unknown -mattr='+sse4.1' -o - | FileCheck %s -check-prefix=SSE_41

; PR20472 ( http://llvm.org/bugs/show_bug.cgi?id=20472 )
; When sexting a trunc'd vector value, we can't eliminate the zext.
; If we don't have SSE4.1, use punpck.
; If we have SSE4.1, use pmovzx because it combines the load op.
; There may be a better way to do this using pshufb + pmovsx,
; but that is beyond our current codegen capabilities.

define <4 x i32> @trunc_sext(<4 x i16>* %in) {
  %load = load <4 x i16>, <4 x i16>* %in
  %trunc = trunc <4 x i16> %load to <4 x i8>
  %sext = sext <4 x i8> %trunc to <4 x i32>
  ret <4 x i32> %sext

; NO_SSE_41-LABEL: trunc_sext:
; NO_SSE_41: movq (%rdi), %xmm0
; NO_SSE_41-NEXT: punpcklwd %xmm0, %xmm0
; NO_SSE_41-NEXT: pslld $24, %xmm0
; NO_SSE_41-NEXT: psrad $24, %xmm0
; NO_SSE_41-NEXT: retq

; SSE_41-LABEL: trunc_sext:
; SSE_41: pmovzxwd (%rdi), %xmm0
; SSE_41-NEXT: pslld $24, %xmm0
; SSE_41-NEXT: psrad $24, %xmm0
; SSE_41-NEXT: retq
}

