; RUN: opt -passes=vector-combine -mtriple=x86_64-unknown-linux-gnu -mcpu=haswell -S < %s | llc -mtriple=x86_64-unknown-linux-gnu -mcpu=haswell | FileCheck %s
; RUN: opt -passes='default<O3>' -mtriple=x86_64-unknown-linux-gnu -mcpu=haswell -S < %s | llc -mtriple=x86_64-unknown-linux-gnu -mcpu=haswell | FileCheck %s

; Shrink the first load before reconstructing the shuffle tree. Reconstructing
; from the original vector loads hides the single-source shuffles needed for
; load shrinking and produces a larger variable permute with a mask constant.

define <8 x i32> @broadcast_halves_optsize(ptr %x, ptr %y) optsize {
; CHECK-LABEL: broadcast_halves_optsize:
; CHECK:       vmovss (%rdi), %xmm0
; CHECK-NEXT:  vshufps $240, (%rsi), %xmm0, %xmm0
; CHECK-NEXT:  vpermpd $80, %ymm0, %ymm0
; CHECK-NEXT:  retq
  %a = load <4 x i32>, ptr %x, align 1
  %b = load <4 x i32>, ptr %y, align 1
  %ba = shufflevector <4 x i32> %a, <4 x i32> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %bb = shufflevector <4 x i32> %b, <4 x i32> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %r = shufflevector <8 x i32> %ba, <8 x i32> %bb, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8>
  ret <8 x i32> %r
}

define <8 x i32> @broadcast_halves_minsize(ptr %x, ptr %y) optsize minsize {
; CHECK-LABEL: broadcast_halves_minsize:
; CHECK:       vmovss (%rdi), %xmm0
; CHECK-NEXT:  vshufps $240, (%rsi), %xmm0, %xmm0
; CHECK-NEXT:  vpermpd $80, %ymm0, %ymm0
; CHECK-NEXT:  retq
  %a = load <4 x i32>, ptr %x, align 1
  %b = load <4 x i32>, ptr %y, align 1
  %ba = shufflevector <4 x i32> %a, <4 x i32> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %bb = shufflevector <4 x i32> %b, <4 x i32> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %r = shufflevector <8 x i32> %ba, <8 x i32> %bb, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8>
  ret <8 x i32> %r
}

; Load shrinking also preserves the align-and-broadcast sequence in throughput
; mode, avoiding two separate scalar broadcasts and an extra insertion.
define <16 x i16> @broadcast_words(ptr %x, ptr %y) {
; CHECK-LABEL: broadcast_words:
; CHECK:       vpbroadcastw (%rsi), %xmm0
; CHECK-NEXT:  vpalignr $14, (%rdi), %xmm0, %xmm0
; CHECK:       vpbroadcastd %xmm0, %ymm0
; CHECK-NEXT:  retq
  %a = load <8 x i16>, ptr %x, align 16
  %b = load <8 x i16>, ptr %y, align 16
  %ba = shufflevector <8 x i16> %a, <8 x i16> poison, <16 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %bb = shufflevector <8 x i16> %b, <8 x i16> poison, <16 x i32> zeroinitializer
  %r = shufflevector <16 x i16> %ba, <16 x i16> %bb, <16 x i32> <i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16>
  ret <16 x i16> %r
}
