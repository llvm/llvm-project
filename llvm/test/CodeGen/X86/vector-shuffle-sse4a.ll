; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+ssse3,+sse4a | FileCheck %s --check-prefix=ALL --check-prefix=BTVER1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+sse4a | FileCheck %s --check-prefix=ALL --check-prefix=BTVER2

;
; EXTRQI
;

; A length of zero is equivalent to a bit length of 64.
define <2 x i64> @extrqi_len0_idx0(<2 x i64> %a) {
; ALL-LABEL: extrqi_len0_idx0:
; ALL:       # BB#0:
; ALL-NEXT:    extrq {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5,6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %a, i8 0, i8 0)
  ret <2 x i64> %1
}

define <2 x i64> @extrqi_len8_idx16(<2 x i64> %a) {
; ALL-LABEL: extrqi_len8_idx16:
; ALL:       # BB#0:
; ALL-NEXT:    extrq {{.*#+}} xmm0 = xmm0[2],zero,zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %a, i8 8, i8 16)
  ret <2 x i64> %1
}

; If the length + index exceeds the bottom 64 bits the result is undefined.
define <2 x i64> @extrqi_len32_idx48(<2 x i64> %a) {
; ALL-LABEL: extrqi_len32_idx48:
; ALL:       # BB#0:
; ALL-NEXT:    extrq {{.*#+}} xmm0 = xmm0[u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %a, i8 32, i8 48)
  ret <2 x i64> %1
}

define <16 x i8> @shuf_0zzzuuuuuuuuuuuu(<16 x i8> %a0) {
; BTVER1-LABEL: shuf_0zzzuuuuuuuuuuuu:
; BTVER1:       # BB#0:
; BTVER1-NEXT:    extrq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    retq
;
; BTVER2-LABEL: shuf_0zzzuuuuuuuuuuuu:
; BTVER2:       # BB#0:
; BTVER2-NEXT:    vpmovzxbq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,zero,zero,zero,zero,xmm0[1],zero,zero,zero,zero,zero,zero,zero
; BTVER2-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 16, i32 16, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %s
}

define <16 x i8> @shuf_0zzzzzzz1zzzzzzz(<16 x i8> %a0) {
; BTVER1-LABEL: shuf_0zzzzzzz1zzzzzzz:
; BTVER1:       # BB#0:
; BTVER1-NEXT:    movaps %xmm0, %xmm1
; BTVER1-NEXT:    extrq {{.*#+}} xmm1 = xmm1[1],zero,zero,zero,zero,zero,zero,zero,xmm1[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    extrq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; BTVER1-NEXT:    retq
;
; BTVER2-LABEL: shuf_0zzzzzzz1zzzzzzz:
; BTVER2:       # BB#0:
; BTVER2-NEXT:    vpmovzxbq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,zero,zero,zero,zero,xmm0[1],zero,zero,zero,zero,zero,zero,zero
; BTVER2-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 1, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <16 x i8> %s
}

define <16 x i8> @shuf_2zzzzzzz3zzzzzzz(<16 x i8> %a0) {
; BTVER1-LABEL: shuf_2zzzzzzz3zzzzzzz:
; BTVER1:       # BB#0:
; BTVER1-NEXT:    movaps %xmm0, %xmm1
; BTVER1-NEXT:    extrq {{.*#+}} xmm1 = xmm1[3],zero,zero,zero,zero,zero,zero,zero,xmm1[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    extrq {{.*#+}} xmm0 = xmm0[2],zero,zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; BTVER1-NEXT:    retq
;
; BTVER2-LABEL: shuf_2zzzzzzz3zzzzzzz:
; BTVER2:       # BB#0:
; BTVER2-NEXT:    vpsrld $16, %xmm0, %xmm0
; BTVER2-NEXT:    vpmovzxbq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,zero,zero,zero,zero,xmm0[1],zero,zero,zero,zero,zero,zero,zero
; BTVER2-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 2, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 3, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <16 x i8> %s
}

define <16 x i8> @shuf_01zzuuuuuuuuuuuu(<16 x i8> %a0) {
; BTVER1-LABEL: shuf_01zzuuuuuuuuuuuu:
; BTVER1:       # BB#0:
; BTVER1-NEXT:    extrq {{.*#+}} xmm0 = xmm0[0,1],zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    retq
;
; BTVER2-LABEL: shuf_01zzuuuuuuuuuuuu:
; BTVER2:       # BB#0:
; BTVER2-NEXT:    vpmovzxwq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero
; BTVER2-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %s
}

define <16 x i8> @shuf_01zzzzzz23zzzzzz(<16 x i8> %a0) {
; BTVER1-LABEL: shuf_01zzzzzz23zzzzzz:
; BTVER1:       # BB#0:
; BTVER1-NEXT:    movaps %xmm0, %xmm1
; BTVER1-NEXT:    extrq {{.*#+}} xmm1 = xmm1[2,3],zero,zero,zero,zero,zero,zero,xmm1[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    extrq {{.*#+}} xmm0 = xmm0[0,1],zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; BTVER1-NEXT:    retq
;
; BTVER2-LABEL: shuf_01zzzzzz23zzzzzz:
; BTVER2:       # BB#0:
; BTVER2-NEXT:    vpmovzxwq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero
; BTVER2-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 2, i32 3, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <16 x i8> %s
}

define <16 x i8> @shuf_1zzzuuuuuuuuuuuu(<16 x i8> %a0) {
; ALL-LABEL: shuf_1zzzuuuuuuuuuuuu:
; ALL:       # BB#0:
; ALL-NEXT:    extrq {{.*#+}} xmm0 = xmm0[1],zero,zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> zeroinitializer, <16 x i32> <i32 1, i32 16, i32 16, i32 16, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %s
}

define <8 x i16> @shuf_1zzzuuuu(<8 x i16> %a0) {
; ALL-LABEL: shuf_1zzzuuuu:
; ALL:       # BB#0:
; ALL-NEXT:    extrq {{.*#+}} xmm0 = xmm0[2,3],zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> zeroinitializer, <8 x i32> <i32 1, i32 8, i32 8, i32 8, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

define <8 x i16> @shuf_12zzuuuu(<8 x i16> %a0) {
; ALL-LABEL: shuf_12zzuuuu:
; ALL:       # BB#0:
; ALL-NEXT:    extrq {{.*#+}} xmm0 = xmm0[2,3,4,5],zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> zeroinitializer, <8 x i32> <i32 1, i32 2, i32 8, i32 8, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

define <8 x i16> @shuf_012zuuuu(<8 x i16> %a0) {
; ALL-LABEL: shuf_012zuuuu:
; ALL:       # BB#0:
; ALL-NEXT:    extrq {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],zero,zero,xmm0[u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

define <8 x i16> @shuf_0zzz1zzz(<8 x i16> %a0) {
; BTVER1-LABEL: shuf_0zzz1zzz:
; BTVER1:       # BB#0:
; BTVER1-NEXT:    movaps %xmm0, %xmm1
; BTVER1-NEXT:    extrq {{.*#+}} xmm1 = xmm1[2,3],zero,zero,zero,zero,zero,zero,xmm1[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    extrq {{.*#+}} xmm0 = xmm0[0,1],zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
; BTVER1-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; BTVER1-NEXT:    retq
;
; BTVER2-LABEL: shuf_0zzz1zzz:
; BTVER2:       # BB#0:
; BTVER2-NEXT:    vpmovzxwq {{.*#+}} xmm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero
; BTVER2-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> zeroinitializer, <8 x i32> <i32 0, i32 8, i32 8, i32 8, i32 1, i32 8, i32 8, i32 8>
  ret <8 x i16> %s
}

define <4 x i32> @shuf_0z1z(<4 x i32> %a0) {
; BTVER1-LABEL: shuf_0z1z:
; BTVER1:       # BB#0:
; BTVER1-NEXT:    pxor %xmm1, %xmm1
; BTVER1-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; BTVER1-NEXT:    retq
;
; BTVER2-LABEL: shuf_0z1z:
; BTVER2:       # BB#0:
; BTVER2-NEXT:    vpmovzxdq {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero
; BTVER2-NEXT:    retq
  %s = shufflevector <4 x i32> %a0, <4 x i32> zeroinitializer, <4 x i32> <i32 0, i32 4, i32 1, i32 4>
  ret <4 x i32> %s
}

;
; INSERTQI
;

; A length of zero is equivalent to a bit length of 64.
define <2 x i64> @insertqi_len0_idx0(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: insertqi_len0_idx0:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm1[0,1,2,3,4,5,6,7],xmm0[u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %a, <2 x i64> %b, i8 0, i8 0)
  ret <2 x i64> %1
}

define <2 x i64> @insertqi_len8_idx16(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: insertqi_len8_idx16:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0,1],xmm1[0],xmm0[3,4,5,6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %a, <2 x i64> %b, i8 8, i8 16)
  ret <2 x i64> %1
}

; If the length + index exceeds the bottom 64 bits the result is undefined
define <2 x i64> @insertqi_len32_idx48(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: insertqi_len32_idx48:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %a, <2 x i64> %b, i8 32, i8 48)
  ret <2 x i64> %1
}

define <16 x i8> @shuf_0_0_2_3_uuuu_uuuu_uuuu(<16 x i8> %a0, <16 x i8> %a1) {
; ALL-LABEL: shuf_0_0_2_3_uuuu_uuuu_uuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0,0,2,3,4,5,6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> %a1, <16 x i32> <i32 0, i32 0, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %s
}

define <16 x i8> @shuf_0_16_2_3_uuuu_uuuu_uuuu(<16 x i8> %a0, <16 x i8> %a1) {
; ALL-LABEL: shuf_0_16_2_3_uuuu_uuuu_uuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[2,3,4,5,6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> %a1, <16 x i32> <i32 0, i32 16, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %s
}

define <16 x i8> @shuf_16_1_2_3_uuuu_uuuu_uuuu(<16 x i8> %a0, <16 x i8> %a1) {
; ALL-LABEL: shuf_16_1_2_3_uuuu_uuuu_uuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm1[0],xmm0[1,2,3,4,5,6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <16 x i8> %a0, <16 x i8> %a1, <16 x i32> <i32 16, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %s
}

define <8 x i16> @shuf_0823uuuu(<8 x i16> %a0, <8 x i16> %a1) {
; ALL-LABEL: shuf_0823uuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0,1],xmm1[0,1],xmm0[4,5,6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> %a1, <8 x i32> <i32 0, i32 8, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

define <8 x i16> @shuf_0183uuuu(<8 x i16> %a0, <8 x i16> %a1) {
; ALL-LABEL: shuf_0183uuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[0,1],xmm0[6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> %a1, <8 x i32> <i32 0, i32 1, i32 8, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

define <8 x i16> @shuf_0128uuuu(<8 x i16> %a0, <8 x i16> %a1) {
; ALL-LABEL: shuf_0128uuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[0,1],xmm0[u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> %a1, <8 x i32> <i32 0, i32 1, i32 2, i32 8, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

define <8 x i16> @shuf_0893uuuu(<8 x i16> %a0, <8 x i16> %a1) {
; ALL-LABEL: shuf_0893uuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0,1],xmm1[0,1,2,3],xmm0[6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> %a1, <8 x i32> <i32 0, i32 8, i32 9, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

define <8 x i16> @shuf_089Auuuu(<8 x i16> %a0, <8 x i16> %a1) {
; ALL-LABEL: shuf_089Auuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0,1],xmm1[0,1,2,3,4,5],xmm0[u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> %a1, <8 x i32> <i32 0, i32 8, i32 9, i32 10, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

define <8 x i16> @shuf_089uuuuu(<8 x i16> %a0, <8 x i16> %a1) {
; ALL-LABEL: shuf_089uuuuu:
; ALL:       # BB#0:
; ALL-NEXT:    insertq {{.*#+}} xmm0 = xmm0[0,1],xmm1[0,1,2,3],xmm0[6,7,u,u,u,u,u,u,u,u]
; ALL-NEXT:    retq
  %s = shufflevector <8 x i16> %a0, <8 x i16> %a1, <8 x i32> <i32 0, i32 8, i32 9, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %s
}

;
; Special Cases
;

; Out of range.
define <16 x i8> @shuffle_8_18_uuuuuuuuuuuuuu(<16 x i8> %a, <16 x i8> %b) {
; BTVER1-LABEL: shuffle_8_18_uuuuuuuuuuuuuu:
; BTVER1:       # BB#0:
; BTVER1-NEXT:    psrld $16, %xmm1
; BTVER1-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; BTVER1-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; BTVER1-NEXT:    retq
;
; BTVER2-LABEL: shuffle_8_18_uuuuuuuuuuuuuu:
; BTVER2:       # BB#0:
; BTVER2-NEXT:    vpsrld $16, %xmm1, %xmm1
; BTVER2-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; BTVER2-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; BTVER2-NEXT:    retq
  %1 = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 8, i32 18, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %1
}

declare <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64>, i8, i8) nounwind
declare <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64>, <2 x i64>, i8, i8) nounwind
