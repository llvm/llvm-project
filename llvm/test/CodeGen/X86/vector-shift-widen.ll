; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+ssse3,-avx,-avx2 | FileCheck %s --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,-avx2 | FileCheck %s --check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=AVX2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2,+sse4a | FileCheck %s --check-prefix=ZNVER1
;
; Check the permutation of a variable shift with i8 vector into a widened shift.
;

; Transform only occurs on SSSE3 because operand is not a shuffle, and shift
; amounts cannot be rearranged to quads. Not checking the correctness of
; untransformed variants here as they are covered by other vector shift checks.
define <16 x i8> @shl_v16i8(<16 x i8> %a) {
; SSSE3-LABEL: shl_v16i8:
; SSSE3:       # %bb.0:
; SSSE3-NEXT:    movdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm1 # xmm1 = [8,1,2,12,4,5,6,7,0,9,10,11,3,13,14,15]
; SSSE3-NEXT:    pshufb %xmm1, %xmm0
; SSSE3-NEXT:    pmullw {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # [1,4,256,256,8,256,16,32]
; SSSE3-NEXT:    pshufb %xmm1, %xmm0
; SSSE3-NEXT:    pand {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: shl_v16i8:
; AVX:         # %bb.0:
; AVX-NOT:       pshufb
; AVX-NOT:       vpshufb
; AVX:           retq
;
; AVX2-LABEL: shl_v16i8:
; AVX2:        # %bb.0:
; AVX2-NOT:      pshufb
; AVX2-NOT:      vpshufb
; AVX2:          retq
  %shift = shl <16 x i8> %a, <i8 3, i8 0, i8 2, i8 4, i8 undef, i8 undef, i8 undef, i8 undef, i8 0, i8 3, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 5>
  ret <16 x i8> %shift
}

define <16 x i8> @lshr_v16i8(<16 x i8> %a) {
; SSSE3-LABEL: lshr_v16i8:
; SSSE3:       # %bb.0:
; SSSE3-NEXT:    pshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # xmm0 = xmm0[2,1,4,3,6,5,8,7,10,9,12,11,14,13,0,15]
; SSSE3-NEXT:    pmulhuw {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # [16384,2048,8192,16384,32768,8192,2048,4096]
; SSSE3-NEXT:    pshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # xmm0 = xmm0[14,1,0,3,2,5,4,7,6,9,8,11,10,13,12,15]
; SSSE3-NEXT:    pand {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: lshr_v16i8:
; AVX:         # %bb.0:
; AVX-NOT:       pshufb
; AVX-NOT:       vpshufb
; AVX:           retq
;
; AVX2-LABEL: lshr_v16i8:
; AVX2:        # %bb.0:
; AVX2-NOT:      pshufb
; AVX2-NOT:      vpshufb
; AVX2:          retq
  %shift = lshr <16 x i8> %a, <i8 4, i8 2, i8 2, i8 5, i8 5, i8 3, i8 3, i8 2, i8 2, i8 1, i8 1, i8 3, i8 3, i8 5, i8 5, i8 4>
  ret <16 x i8> %shift
}

define <16 x i8> @ashr_v16i8(<16 x i8> %a) {
; SSSE3-LABEL: ashr_v16i8:
; SSSE3:       # %bb.0:
; SSSE3-NEXT:    pshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # xmm0 = xmm0[0,12,2,3,4,9,11,7,8,13,10,6,1,14,5,15]
; SSSE3-NEXT:    pmulhuw {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # [16384,8192,512,8192,4096,1024,32768,2048]
; SSSE3-NEXT:    pshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # xmm0 = xmm0[0,12,2,3,4,14,11,7,8,5,10,6,1,9,13,15]
; SSSE3-NEXT:    pand {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; SSSE3-NEXT:    movdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm1 # xmm1 = [32,64,16,16,1,4,2,16,8,1,u,16,32,8,64,4]
; SSSE3-NEXT:    pxor %xmm1, %xmm0
; SSSE3-NEXT:    psubb %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: ashr_v16i8:
; AVX:         # %bb.0:
; AVX-NOT:       pshufb
; AVX-NOT:       vpshufb
; AVX:           retq
;
; AVX2-LABEL: ashr_v16i8:
; AVX2:        # %bb.0:
; AVX2-NOT:      pshufb
; AVX2-NOT:      vpshufb
; AVX2:          retq
  %shift = ashr <16 x i8> %a, <i8 2, i8 1, i8 3, i8 3, i8 7, i8 5, i8 6, i8 3, i8 4, i8 7, i8 undef, i8 3, i8 2, i8 4, i8 1, i8 5>
  ret <16 x i8> %shift
}

; Shift amounts cannot be paired.
define <16 x i8> @not_shl_v16i8(<16 x i8> %a) {
; SSSE3-LABEL: not_shl_v16i8:
; SSSE3:       # %bb.0:
; SSSE3-NOT:     pshufb
; SSSE3-NOT:     vpshufb
; SSSE3:         retq
;
; AVX-LABEL: not_shl_v16i8:
; AVX:         # %bb.0:
; AVX-NOT:       pshufb
; AVX-NOT:       vpshufb
; AVX:           retq
;
; AVX2-LABEL: not_shl_v16i8:
; AVX2:        # %bb.0:
; AVX2-NOT:      pshufb
; AVX2-NOT:      vpshufb
; AVX2:          retq
  %shift = shl <16 x i8> %a, <i8 2, i8 1, i8 3, i8 0, i8 7, i8 5, i8 6, i8 4, i8 2, i8 1, i8 3, i8 0, i8 7, i8 5, i8 6, i8 5>
  ret <16 x i8> %shift
}

; Right shift amounts containing zero and cannot form quads.
define <16 x i8> @not_lshr_v16i8(<16 x i8> %a) {
; SSSE3-LABEL: not_lshr_v16i8:
; SSSE3:       # %bb.0:
; SSSE3-NOT:     pshufb
; SSSE3-NOT:     vpshufb
; SSSE3:         retq
;
; AVX-LABEL: not_lshr_v16i8:
; AVX:         # %bb.0:
; AVX-NOT:       pshufb
; AVX-NOT:       vpshufb
; AVX:           retq
;
; AVX2-LABEL: not_lshr_v16i8:
; AVX2:        # %bb.0:
; AVX2-NOT:      pshufb
; AVX2-NOT:      vpshufb
; AVX2:          retq
  %shift = lshr <16 x i8> %a, <i8 4, i8 2, i8 2, i8 5, i8 5, i8 3, i8 3, i8 2, i8 2, i8 1, i8 1, i8 0, i8 0, i8 5, i8 5, i8 4>
  ret <16 x i8> %shift
}

; Shift cannot form quads and operand is not shuffle, only transform on SSSE3.
define <32 x i8> @shl_v32i8(<32 x i8> %a) {
; SSSE3-LABEL: shl_v32i8:
; SSSE3:       # %bb.0:
; SSSE3-NEXT:    movdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm2 # xmm2 = [0,2,1,3,6,5,4,7,8,9,12,11,10,13,14,15]
; SSSE3-NEXT:    pshufb %xmm2, %xmm0
; SSSE3-NEXT:    movdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm3 # xmm3 = [1,4,8,2,16,32,64,16]
; SSSE3-NEXT:    pmullw %xmm3, %xmm0
; SSSE3-NEXT:    pshufb %xmm2, %xmm0
; SSSE3-NEXT:    movdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm4 # xmm4 = [255,252,255,252,254,248,248,254,240,240,192,224,224,192,240,240]
; SSSE3-NEXT:    pand %xmm4, %xmm0
; SSSE3-NEXT:    pshufb %xmm2, %xmm1
; SSSE3-NEXT:    pmullw %xmm3, %xmm1
; SSSE3-NEXT:    pshufb %xmm2, %xmm1
; SSSE3-NEXT:    pand %xmm4, %xmm1
; SSSE3-NEXT:    retq
;
; AVX-LABEL: shl_v32i8:
; AVX:         # %bb.0:
; AVX-NOT:       pshufb
; AVX-NOT:       vpshufb
; AVX:           retq
;
; AVX2-LABEL: shl_v32i8:
; AVX2:        # %bb.0:
; AVX2-NOT:      pshufb
; AVX2-NOT:      vpshufb
; AVX2:          retq
  %shift = shl <32 x i8> %a, <i8 0, i8 2, i8 0, i8 2, i8 1, i8 3, i8 3, i8 1, i8 4, i8 4, i8 6, i8 5, i8 5, i8 6, i8 4, i8 4,
                              i8 0, i8 2, i8 0, i8 2, i8 1, i8 3, i8 3, i8 1, i8 4, i8 4, i8 6, i8 5, i8 5, i8 6, i8 4, i8 4>
  ret <32 x i8> %shift
}

; For quads only testing on AVX2 as it has vps**vd.
define <32 x i8> @shl_v32i8_quad(<32 x i8> %a) {
; AVX2-LABEL: shl_v32i8_quad:
; AVX2:        # %bb.0:
; AVX2-NEXT:     vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0 # ymm0 = ymm0[0,5,13,9,3,6,12,11,2,4,10,14,1,7,8,15,25,29,18,22,24,28,19,23,17,21,26,30,16,20,27,31]
; AVX2-NEXT:     vpsllvd {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:     vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0 # ymm0 = ymm0[0,12,8,4,9,1,5,13,14,3,10,7,6,2,11,15,28,24,18,22,29,25,19,23,20,16,26,30,21,17,27,31]
; AVX2-NEXT:     vpand {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:     retq
  %shift = shl <32 x i8> %a, <i8 0, i8 2, i8 4, i8 6, i8 4, i8 0, i8 6, i8 2, i8 2, i8 0, i8 4, i8 6, i8 6, i8 0, i8 4, i8 2,
                              i8 1, i8 3, i8 5, i8 7, i8 1, i8 3, i8 5, i8 7, i8 7, i8 5, i8 3, i8 1, i8 7, i8 5, i8 3, i8 1>
  ret <32 x i8> %shift
}

define <32 x i8> @lshr_v32i8_quad(<32 x i8> %a) {
; AVX2-LABEL: lshr_v32i8_quad:
; AVX2:        # %bb.0:
; AVX2-NEXT:     vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0 # ymm0 = ymm0[0,5,13,9,3,6,12,11,2,4,10,14,1,7,8,15,25,29,18,22,24,28,19,23,17,21,26,30,16,20,27,31]
; AVX2-NEXT:     vpsrlvd {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:     vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0 # ymm0 = ymm0[0,12,8,4,9,1,5,13,14,3,10,7,6,2,11,15,28,24,18,22,29,25,19,23,20,16,26,30,21,17,27,31]
; AVX2-NEXT:     vpand {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:     retq
  %shift = lshr <32 x i8> %a, <i8 0, i8 2, i8 4, i8 6, i8 4, i8 0, i8 6, i8 2, i8 2, i8 0, i8 4, i8 6, i8 6, i8 0, i8 4, i8 2,
                               i8 1, i8 3, i8 5, i8 7, i8 1, i8 3, i8 5, i8 7, i8 7, i8 5, i8 3, i8 1, i8 7, i8 5, i8 3, i8 1>
  ret <32 x i8> %shift
}

; Disabling the transform for AMD Zen because it can schedule two vpmullw 2
; cycles faster compared to Intel.
define <32 x i8> @ashr_v32i8_quad(<32 x i8> %a) {
; AVX2-LABEL: ashr_v32i8_quad:
; AVX2:        # %bb.0:
; AVX2-NEXT:     vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0 # ymm0 = ymm0[0,5,13,9,3,6,12,11,2,4,10,14,1,7,8,15,25,29,18,22,24,28,19,23,17,21,26,30,16,20,27,31]
; AVX2-NEXT:     vpsrlvd {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:     vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0 # ymm0 = ymm0[0,12,8,4,9,1,5,13,14,3,10,7,6,2,11,15,28,24,18,22,29,25,19,23,20,16,26,30,21,17,27,31]
; AVX2-NEXT:     vpand {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm0, %ymm0
; AVX2-NEXT:     vmovdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %ymm1 # ymm1 = [128,32,8,2,8,128,2,32,32,128,8,2,2,128,8,32,64,16,4,1,64,16,4,1,1,4,16,64,1,4,16,64]
; AVX2-NEXT:     vpxor %ymm1, %ymm0, %ymm0
; AVX2-NEXT:     vpsubb %ymm1, %ymm0, %ymm0
; AVX2-NEXT:     retq
;
; ZNVER1-LABEL: ashr_v32i8_quad:
; ZNVER1:      # %bb.0:
; ZNVER1-NOT:    pshufb
; ZNVER1-NOT:    vpshufb
; ZNVER1:        retq
  %shift = ashr <32 x i8> %a, <i8 0, i8 2, i8 4, i8 6, i8 4, i8 0, i8 6, i8 2, i8 2, i8 0, i8 4, i8 6, i8 6, i8 0, i8 4, i8 2,
                               i8 1, i8 3, i8 5, i8 7, i8 1, i8 3, i8 5, i8 7, i8 7, i8 5, i8 3, i8 1, i8 7, i8 5, i8 3, i8 1>
  ret <32 x i8> %shift
}

; Shift amounts cannot be paired in lane.
define <32 x i8> @not_shl_v32i8(<32 x i8> %a) {
; SSSE3-LABEL: not_shl_v32i8:
; SSSE3:       # %bb.0:
; SSSE3-NOT:     pshufb
; SSSE3-NOT:     vpshufb
; SSSE3:         retq
;
; AVX-LABEL: not_shl_v32i8:
; AVX:         # %bb.0:
; AVX-NOT:       pshufb
; AVX-NOT:       vpshufb
; AVX:           retq
;
; AVX2-LABEL: not_shl_v32i8:
; AVX2:        # %bb.0:
; AVX2-NOT:      pshufb
; AVX2-NOT:      vpshufb
; AVX2:          retq
  %shift = shl <32 x i8> %a, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 3,
                              i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 3, i8 3, i8 3>
  ret <32 x i8> %shift
}

; Always transform if operand is shuffle and shift amounts can be paired.
define <16 x i8> @lshr_shuffle_v16i8(<16 x i8> %a) {
; SSSE3-LABEL: lshr_shuffle_v16i8:
; SSSE3:       # %bb.0:
; SSSE3-NEXT:    pshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # xmm0 = xmm0[0,8,4,12,1,9,5,13,2,10,6,14,3,11,7,15]
; SSSE3-NEXT:    pmulhuw {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # [32768,16384,16384,8192,8192,4096,4096,2048]
; SSSE3-NEXT:    pshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0 # xmm0 = xmm0[0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15]
; SSSE3-NEXT:    pand {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; SSSE3-NEXT:    movdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm1 # xmm1 = [64,32,64,32,32,16,32,16,16,8,16,8,8,4,8,4]
; SSSE3-NEXT:    pxor %xmm1, %xmm0
; SSSE3-NEXT:    psubb %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: lshr_shuffle_v16i8:
; AVX:       # %bb.0:
; AVX-NEXT:    vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # xmm0 = xmm0[0,8,4,12,1,9,5,13,2,10,6,14,3,11,7,15]
; AVX-NEXT:    vpmulhuw {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # [32768,16384,16384,8192,8192,4096,4096,2048]
; AVX-NEXT:    vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # xmm0 = xmm0[0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15]
; AVX-NEXT:    vpand {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vmovdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm1 # xmm1 = [64,32,64,32,32,16,32,16,16,8,16,8,8,4,8,4]
; AVX-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsubb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
;
; AVX2-LABEL: lshr_shuffle_v16i8:
; AVX2:       # %bb.0:
; AVX2-NEXT:    vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # xmm0 = xmm0[0,8,4,12,1,9,5,13,2,10,6,14,3,11,7,15]
; AVX2-NEXT:    vpmulhuw {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # [32768,16384,16384,8192,8192,4096,4096,2048]
; AVX2-NEXT:    vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # xmm0 = xmm0[0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15]
; AVX2-NEXT:    vpand {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0
; AVX2-NEXT:    vmovdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm1 # xmm1 = [64,32,64,32,32,16,32,16,16,8,16,8,8,4,8,4]
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpsubb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    retq
;
; ZNVER1-LABEL: lshr_shuffle_v16i8:
; ZNVER1:       # %bb.0:
; ZNVER1-NEXT:    vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # xmm0 = xmm0[0,8,4,12,1,9,5,13,2,10,6,14,3,11,7,15]
; ZNVER1-NEXT:    vpmulhuw {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # [32768,16384,16384,8192,8192,4096,4096,2048]
; ZNVER1-NEXT:    vpshufb {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0 # xmm0 = xmm0[0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15]
; ZNVER1-NEXT:    vpand {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm0, %xmm0
; ZNVER1-NEXT:    vmovdqa {{\.LCPI[0-9]+_[0-9]+}}(%rip), %xmm1 # xmm1 = [64,32,64,32,32,16,32,16,16,8,16,8,8,4,8,4]
; ZNVER1-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; ZNVER1-NEXT:    vpsubb %xmm1, %xmm0, %xmm0
; ZNVER1-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  %shift = ashr <16 x i8> %shuffle, <i8 1, i8 2, i8 1, i8 2, i8 2, i8 3, i8 2, i8 3, i8 3, i8 4, i8 3, i8 4, i8 4, i8 5, i8 4, i8 5>
  ret <16 x i8> %shift
}
