; RUN: llc -march=x86-64 -mattr=sse4.1 < %s | FileCheck %s -check-prefix=SSE4
; RUN: llc -march=x86-64 -mattr=avx < %s | FileCheck %s -check-prefix=AVX1
; RUN: llc -march=x86-64 -mattr=avx2 < %s | FileCheck %s -check-prefix=AVX2

define <16 x i16> @split16(<16 x i16> %a, <16 x i16> %b, <16 x i8> %__mask) {
; SSE4-LABEL: split16:
; SSE4: pminuw
; SSE4: pminuw
; SSE4: ret
; AVX1-LABEL: split16:
; AVX1: vpminuw
; AVX1: vpminuw
; AVX1: ret
; AVX2-LABEL: split16:
; AVX2: vpminuw
; AVX2: ret
  %1 = icmp ult <16 x i16> %a, %b
  %2 = select <16 x i1> %1, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %2
}

define <32 x i16> @split32(<32 x i16> %a, <32 x i16> %b, <32 x i8> %__mask) {
; SSE4-LABEL: split32:
; SSE4: pminuw
; SSE4: pminuw
; SSE4: pminuw
; SSE4: pminuw
; SSE4: ret
; AVX1-LABEL: split32:
; AVX1: vpminuw
; AVX1: vpminuw
; AVX1: vpminuw
; AVX1: vpminuw
; AVX1: ret
; AVX2-LABEL: split32:
; AVX2: vpminuw
; AVX2: vpminuw
; AVX2: ret
  %1 = icmp ult <32 x i16> %a, %b
  %2 = select <32 x i1> %1, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %2
}

; PR19492
define i128 @split128(<2 x i128> %a, <2 x i128> %b) {
; SSE4-LABEL: split128:
; SSE4: addq
; SSE4: adcq
; SSE4: addq
; SSE4: adcq
; SSE4: addq
; SSE4: adcq
; SSE4: ret
; AVX1-LABEL: split128:
; AVX1: addq
; AVX1: adcq
; AVX1: addq
; AVX1: adcq
; AVX1: addq
; AVX1: adcq
; AVX1: ret
; AVX2-LABEL: split128:
; AVX2: addq
; AVX2: adcq
; AVX2: addq
; AVX2: adcq
; AVX2: addq
; AVX2: adcq
; AVX2: ret
  %add = add nsw <2 x i128> %a, %b
  %rdx.shuf = shufflevector <2 x i128> %add, <2 x i128> undef, <2 x i32> <i32 undef, i32 0>
  %bin.rdx = add <2 x i128> %add, %rdx.shuf
  %e = extractelement <2 x i128> %bin.rdx, i32 1
  ret i128 %e
}
