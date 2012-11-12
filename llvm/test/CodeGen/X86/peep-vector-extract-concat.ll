; RUN: llc < %s -mtriple=x86_64-linux -mattr=+sse2,-sse41 | FileCheck %s
; CHECK: pshufd $3, %xmm0, %xmm0

; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+sse2,-sse41 | FileCheck %s -check-prefix=WIN64
; %a is passed indirectly on Win64.
; WIN64: movss   12(%rcx), %xmm0

define float @foo(<8 x float> %a) nounwind {
  %c = extractelement <8 x float> %a, i32 3
  ret float %c
}
