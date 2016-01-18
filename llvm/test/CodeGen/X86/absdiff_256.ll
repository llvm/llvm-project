; RUN: llc < %s -mtriple=x86_64-unknown-unknown  | FileCheck %s

declare <16 x i16> @llvm.sabsdiff.v16i16(<16 x i16>, <16 x i16>)

define <16 x i16> @test_sabsdiff_v16i16_expand(<16 x i16> %a1, <16 x i16> %a2) {
; CHECK-LABEL: test_sabsdiff_v16i16_expand:
; CHECK:       # BB#0:
; CHECK:         psubw
; CHECK:         pxor
; CHECK:         pcmpgtw
; CHECK:         movdqa
; CHECK:         pandn
; CHECK:         pxor
; CHECK:         psubw
; CHECK:         pcmpeqd
; CHECK:         pxor
; CHECK:         pandn
; CHECK:         por
; CHECK:         pcmpgtw
; CHECK-DAG:     psubw {{%xmm[0-9]+}}, [[SRC:%xmm[0-9]+]]
; CHECK-DAG:     pxor {{%xmm[0-9]+}}, [[DST:%xmm[0-9]+]]
; CHECK:         pandn [[SRC]], [[DST]]
; CHECK:         por
; CHECK:         movdqa
; CHECK:         retq
  %1 = call <16 x i16> @llvm.sabsdiff.v16i16(<16 x i16> %a1, <16 x i16> %a2)
  ret <16 x i16> %1
}

