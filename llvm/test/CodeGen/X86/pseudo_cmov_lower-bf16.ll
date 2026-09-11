; RUN: llc < %s -mtriple=x86_64 -mattr=+avx512bf16,+avx512vl | FileCheck %s
; RUN: llc < %s -mtriple=x86_64 -mattr=+avx10.2-512 | FileCheck %s

; https://github.com/llvm/llvm-project/issues/222673
; CHECK-LABEL: pr222673:
; CHECK:       testb $1, %dil
; CHECK-NEXT:  jne
; CHECK-NOT:   jne
define <8 x bfloat> @pr222673(i1 %0) {
  %2 = select i1 %0, <8 x bfloat> splat (bfloat 1.000000e+00), <8 x bfloat> zeroinitializer
  ret <8 x bfloat> %2
}

; CHECK-LABEL: select_v8bf16:
; CHECK:       jne
; CHECK-NOT:   jne
define <8 x bfloat> @select_v8bf16(<8 x bfloat> %a, <8 x bfloat> %b, i1 zeroext %sign) {
  %sel = select i1 %sign, <8 x bfloat> %a, <8 x bfloat> %b
  ret <8 x bfloat> %sel
}

; CHECK-LABEL: select_v16bf16:
; CHECK:       jne
; CHECK-NOT:   jne
define <16 x bfloat> @select_v16bf16(<16 x bfloat> %a, <16 x bfloat> %b, i1 zeroext %sign) {
  %sel = select i1 %sign, <16 x bfloat> %a, <16 x bfloat> %b
  ret <16 x bfloat> %sel
}

; CHECK-LABEL: select_v32bf16:
; CHECK:       jne
; CHECK-NOT:   jne
define <32 x bfloat> @select_v32bf16(<32 x bfloat> %a, <32 x bfloat> %b, i1 zeroext %sign) {
  %sel = select i1 %sign, <32 x bfloat> %a, <32 x bfloat> %b
  ret <32 x bfloat> %sel
}

; Both selects share one condition, so only one branch should remain.
; CHECK-LABEL: select_v8bf16_chain:
; CHECK:       je
; CHECK-NOT:   je
define <8 x bfloat> @select_v8bf16_chain(i32 %v1, <8 x bfloat> %v2, <8 x bfloat> %v3, <8 x bfloat> %v4) {
  %cmp = icmp eq i32 %v1, 0
  %t1 = select i1 %cmp, <8 x bfloat> %v2, <8 x bfloat> %v3
  %t2 = select i1 %cmp, <8 x bfloat> %v3, <8 x bfloat> %v4
  %sub = fsub <8 x bfloat> %t1, %t2
  ret <8 x bfloat> %sub
}
