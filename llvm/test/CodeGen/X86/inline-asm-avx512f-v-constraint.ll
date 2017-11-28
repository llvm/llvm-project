; RUN: llc < %s -mtriple x86_64-unknown-linux-gnu -mattr +avx512f | FileCheck %s

define <16 x float> @testzmm_1(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vpternlogd  $0, %zmm1, %zmm0, %zmm0
  %0 = tail call <16 x float> asm "vpternlogd $$0, $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1, <16 x float> %_zmm0)
  ret <16 x float> %0
}

define <16 x float> @testzmm_2(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vpabsq  %zmm1, %zmm0
  %0 = tail call <16 x float> asm "vpabsq $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1)
  ret <16 x float> %0
}


define <16 x float> @testzmm_3(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vpaddd  %zmm1, %zmm1, %zmm0
  %0 = tail call <16 x float> asm "vpaddd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1, <16 x float> %_zmm1)
  ret <16 x float> %0
}


define <16 x float> @testzmm_4(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vpaddq  %zmm1, %zmm1, %zmm0
  %0 = tail call <16 x float> asm "vpaddq $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1, <16 x float> %_zmm1)
  ret <16 x float> %0
}


define <16 x float> @testzmm_5(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vpandd  %zmm1, %zmm1, %zmm0
  %0 = tail call <16 x float> asm "vpandd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1, <16 x float> %_zmm1)
  ret <16 x float> %0
}


define <16 x float> @testzmm_6(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vpandnd %zmm1, %zmm1, %zmm0
  %0 = tail call <16 x float> asm "vpandnd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1, <16 x float> %_zmm1)
  ret <16 x float> %0
}


define <16 x float> @testzmm_7(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vpmaxsd %zmm1, %zmm1, %zmm0
  %0 = tail call <16 x float> asm "vpmaxsd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1, <16 x float> %_zmm1)
  ret <16 x float> %0
}


define <16 x float> @testzmm_8(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vmovups %zmm1, %zmm0
  %0 = tail call <16 x float> asm "vmovups $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1)
  ret <16 x float> %0
}


define <16 x float> @testzmm_9(<16 x float> %_zmm0, <16 x float> %_zmm1) {
entry:
; CHECK: vmovupd %zmm1, %zmm0
  %0 = tail call <16 x float> asm "vmovupd $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<16 x float> %_zmm1)
  ret <16 x float> %0
}

