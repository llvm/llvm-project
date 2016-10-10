; RUN: llc < %s -march x86-64 -mtriple x86_64-unknown-linux-gnu -mattr +avx512vl | FileCheck %s

define <4 x float> @testXMM_1(<4 x float> %_xmm0, i64 %_l) {
entry:
; CHECK: vmovhlps  %xmm17, %xmm16, %xmm16
  %0 = tail call <4 x float> asm "vmovhlps $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{dirflag},~{fpsr},~{flags}"(i64 %_l, <4 x float> %_xmm0)
  ret <4 x float> %0
}

define <4 x float> @testXMM_2(<4 x float> %_xmm0, i64 %_l) {
entry:
; CHECK: vmovapd %xmm16, %xmm16
  %0 = tail call <4 x float> asm "vmovapd $1, $0", "=v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{dirflag},~{fpsr},~{flags}"(i64 %_l)
  ret <4 x float> %0
}

define <4 x float> @testXMM_3(<4 x float> %_xmm0, i64 %_l) {
entry:
; CHECK: vminpd  %xmm16, %xmm16, %xmm16
  %0 = tail call <4 x float> asm "vminpd $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{dirflag},~{fpsr},~{flags}"(i64 %_l, i64 %_l)
  ret <4 x float> %0
}

define i64 @testXMM_4(<4 x float> %_xmm0, i64 %_l) {
entry:
; CHECK: vmulsd  %xmm17, %xmm16, %xmm16
  %0 = tail call i64 asm "vmulsd $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{dirflag},~{fpsr},~{flags}"(i64 %_l, <4 x float> %_xmm0)
  ret i64 %0
}

define <4 x float> @testXMM_5(<4 x float> %_xmm0, i64 %_l) {
entry:
; CHECK: vpabsq  %xmm16, %xmm16
  %0 = tail call <4 x float> asm "vpabsq $1, $0", "=v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{dirflag},~{fpsr},~{flags}"(i64 %_l)
  ret <4 x float> %0
}

define <4 x float> @testXMM_6(<4 x float> %_xmm0, i64 %_l) {
entry:
; CHECK: vpandd  %xmm16, %xmm17, %xmm16
  %0 = tail call <4 x float> asm "vpandd $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{dirflag},~{fpsr},~{flags}"(<4 x float> %_xmm0, i64 %_l)
  ret <4 x float> %0
}

define <4 x float> @testXMM_7(<4 x float> %_xmm0, i64 %_l) {
entry:
; CHECK: vpandnd %xmm16, %xmm17, %xmm16
  %0 = tail call <4 x float> asm "vpandnd $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{xmm8},~{xmm9},~{xmm10},~{xmm11},~{xmm12},~{xmm13},~{xmm14},~{xmm15},~{dirflag},~{fpsr},~{flags}"(<4 x float> %_xmm0, i64 %_l)
  ret <4 x float> %0
}

define <8 x float> @testYMM_1(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vmovsldup %ymm16, %ymm16
  %0 = tail call <8 x float> asm "vmovsldup $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testYMM_2(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vmovapd %ymm16, %ymm16
  %0 = tail call <8 x float> asm "vmovapd $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testYMM_3(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vminpd  %ymm16, %ymm16, %ymm16
  %0 = tail call <8 x float> asm "vminpd $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testYMM_4(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vpabsq  %ymm16, %ymm16
  %0 = tail call <8 x float> asm "vpabsq $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testYMM_5(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vpandd  %ymm16, %ymm17, %ymm16
  %0 = tail call <8 x float> asm "vpandd $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM_6(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vpandnd %ymm16, %ymm17, %ymm16
  %0 = tail call <8 x float> asm "vpandnd $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM_7(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vpminud %ymm16, %ymm17, %ymm16
  %0 = tail call <8 x float> asm "vpminud $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM_8(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vpmaxsd %ymm16, %ymm17, %ymm16
  %0 = tail call <8 x float> asm "vpmaxsd $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM_9(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vmovups %ymm16, %ymm16
  %0 = tail call <8 x float> asm "vmovups $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testYMM_10(<8 x float> %_ymm0, <8 x float> %_ymm1) {
entry:
; CHECK: vmovupd %ymm16, %ymm16
  %0 = tail call <8 x float> asm "vmovupd $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{ymm8},~{ymm9},~{ymm10},~{ymm11},~{ymm12},~{ymm13},~{ymm14},~{ymm15},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

