; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

; PR11580
define <3 x float> @addf3(<3 x float> %x) {
; CHECK-LABEL: addf3
; CHECK:       # BB#0:
; CHECK-NEXT:  addps .LCPI0_0(%rip), %xmm0
; CHECK-NEXT:  retq
entry:
  %add = fadd <3 x float> %x, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  ret <3 x float> %add
}

; PR11580
define <4 x float> @cvtf3_f4(<3 x float> %x) {
; CHECK-LABEL: cvtf3_f4
; CHECK:       # BB#0:
; CHECK-NEXT:  retq
entry:
  %extractVec = shufflevector <3 x float> %x, <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  ret <4 x float> %extractVec
}

; PR11580
define <3 x float> @cvtf4_f3(<4 x float> %x) {
; CHECK-LABEL: cvtf4_f3
; CHECK:       # BB#0:
; CHECK-NEXT:  retq
entry:
  %extractVec = shufflevector <4 x float> %x, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  ret <3 x float> %extractVec
}
