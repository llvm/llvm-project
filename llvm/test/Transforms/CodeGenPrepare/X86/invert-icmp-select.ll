; RUN: opt -S -passes='require<profile-summary>,function(codegenprepare)' \
; RUN:   -mtriple=x86_64 -mattr=+avx2 %s | FileCheck %s --check-prefix=AVX2
; RUN: opt -S -passes='require<profile-summary>,function(codegenprepare)' \
; RUN:   -mtriple=x86_64 -mattr=+avx512bw %s | FileCheck %s --check-prefix=AVX512

define <8 x i8> @direct(<8 x i8> %a, <8 x i8> %b, <8 x i8> %x, <8 x i8> %y) {
; AVX2-LABEL: @direct(
; AVX2: %cmp = icmp uge <8 x i8> %a, %b
; AVX2: %sel = select <8 x i1> %cmp, <8 x i8> %y, <8 x i8> %x, !prof [[INVERTED_WEIGHTS:![0-9]+]]
; AVX512-LABEL: @direct(
; AVX512: %cmp = icmp ult <8 x i8> %a, %b
; AVX512: %sel = select <8 x i1> %cmp, <8 x i8> %x, <8 x i8> %y, !prof [[ORIGINAL_WEIGHTS:![0-9]+]]
  %cmp = icmp ult <8 x i8> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i8> %x, <8 x i8> %y, !prof !0
  ret <8 x i8> %sel
}

define <4 x i8> @through_shuffle(<8 x i8> %a, <8 x i8> %b,
                                 <4 x i8> %x, <4 x i8> %y) {
; AVX2-LABEL: @through_shuffle(
; AVX2: %cmp = icmp uge <8 x i8> %a, %b
; AVX2: %cond = shufflevector <8 x i1> %cmp, <8 x i1> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; AVX2: %sel = select <4 x i1> %cond, <4 x i8> %y, <4 x i8> %x
  %cmp = icmp ult <8 x i8> %a, %b
  %cond = shufflevector <8 x i1> %cmp, <8 x i1> poison,
                        <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %sel = select <4 x i1> %cond, <4 x i8> %x, <4 x i8> %y
  ret <4 x i8> %sel
}

define <8 x i8> @not_profitable(<8 x i8> %a, <8 x i8> %b,
                                <8 x i8> %x, <8 x i8> %y) {
; AVX2-LABEL: @not_profitable(
; AVX2: %cmp = icmp eq <8 x i8> %a, %b
; AVX2: %sel = select <8 x i1> %cmp, <8 x i8> %x, <8 x i8> %y
  %cmp = icmp eq <8 x i8> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i8> %x, <8 x i8> %y
  ret <8 x i8> %sel
}

define <8 x i8> @shared_cmp(<8 x i8> %a, <8 x i8> %b,
                            <8 x i8> %x, <8 x i8> %y) {
; AVX2-LABEL: @shared_cmp(
; AVX2: %cmp = icmp ult <8 x i8> %a, %b
; AVX2: %sel1 = select <8 x i1> %cmp, <8 x i8> %x, <8 x i8> %y
; AVX2: %sel2 = select <8 x i1> %cmp, <8 x i8> %sel1, <8 x i8> %a
  %cmp = icmp ult <8 x i8> %a, %b
  %sel1 = select <8 x i1> %cmp, <8 x i8> %x, <8 x i8> %y
  %sel2 = select <8 x i1> %cmp, <8 x i8> %sel1, <8 x i8> %a
  ret <8 x i8> %sel2
}

; AVX2: [[INVERTED_WEIGHTS]] = !{!"branch_weights", i32 90, i32 10}
; AVX512: [[ORIGINAL_WEIGHTS]] = !{!"branch_weights", i32 10, i32 90}

!0 = !{!"branch_weights", i32 10, i32 90}
