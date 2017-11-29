; RUN: llc < %s -mtriple x86_64-unknown-linux-gnu -mattr +avx | FileCheck %s
; RUN: llc < %s -mtriple x86_64-unknown-linux-gnu -mattr +avx512f | FileCheck %s

define <4 x float> @testxmm_1(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: vmovhlps  %xmm1, %xmm0, %xmm0
entry:
  %0 = tail call <4 x float> asm "vmovhlps $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l, <4 x float> %_xmm0)
  ret <4 x float> %0
}

define <4 x float> @testxmm_2(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: movapd  %xmm0, %xmm0
entry:
  %0 = tail call <4 x float> asm "movapd $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l)
  ret <4 x float> %0
}

define <4 x float> @testxmm_3(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: vmovapd %xmm0, %xmm0
entry:
  %0 = tail call <4 x float> asm "vmovapd $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l)
  ret <4 x float> %0
}

define <4 x float> @testxmm_4(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: vmpsadbw  $0, %xmm1, %xmm0, %xmm0
entry:
  %0 = tail call <4 x float> asm "vmpsadbw $$0, $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l, <4 x float> %_xmm0)
  ret <4 x float> %0
}

define <4 x float> @testxmm_5(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: vminpd  %xmm0, %xmm0, %xmm0
entry:
  %0 = tail call <4 x float> asm "vminpd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l, i64 %_l)
  ret <4 x float> %0
}

define i64 @testxmm_6(i64 returned %_l)  {
; CHECK: vmovd %xmm0, %eax
entry:
  tail call void asm sideeffect "vmovd $0, %eax", "v,~{dirflag},~{fpsr},~{flags}"(i64 %_l)
  ret i64 %_l
}

define <4 x float> @testxmm_7(<4 x float> returned %_xmm0) {
; CHECK: vmovmskps %xmm0, %eax
entry:
  tail call void asm sideeffect "vmovmskps $0, %rax", "v,~{dirflag},~{fpsr},~{flags}"(<4 x float> %_xmm0)
  ret <4 x float> %_xmm0
}

define i64 @testxmm_8(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: vmulsd  %xmm1, %xmm0, %xmm0
entry:
  %0 = tail call i64 asm "vmulsd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l, <4 x float> %_xmm0)
  ret i64 %0
}

define <4 x float> @testxmm_9(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: vorpd %xmm1, %xmm0, %xmm0
entry:
  %0 = tail call <4 x float> asm "vorpd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l, <4 x float> %_xmm0)
  ret <4 x float> %0
}

define <4 x float> @testxmm_10(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: pabsb %xmm0, %xmm0
entry:
  %0 = tail call <4 x float> asm "pabsb $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l)
  ret <4 x float> %0
}

define <4 x float> @testxmm_11(<4 x float> %_xmm0, i64 %_l)  {
; CHECK: vpabsd  %xmm0, %xmm0
entry:
  %0 = tail call <4 x float> asm "vpabsd $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(i64 %_l)
  ret <4 x float> %0
}

define <8 x float> @testymm_1(<8 x float> %_ymm0, <8 x float> %_ymm1)  {
; CHECK: vmovsldup %ymm0, %ymm0
entry:
  %0 = tail call <8 x float> asm "vmovsldup $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testymm_2(<8 x float> %_ymm0, <8 x float> %_ymm1)  {
; CHECK: vmovapd %ymm1, %ymm0
entry:
  %0 = tail call <8 x float> asm "vmovapd $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testymm_3(<8 x float> %_ymm0, <8 x float> %_ymm1)  {
; CHECK: vminpd  %ymm1, %ymm0, %ymm0
entry:
  %0 = tail call <8 x float> asm "vminpd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testymm_4(<8 x float> %_ymm0, <8 x float> %_ymm1)  {
; CHECK: vorpd %ymm1, %ymm0, %ymm0
entry:
  %0 = tail call <8 x float> asm "vorpd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testymm(<8 x float> %_ymm0, <8 x float> %_ymm1)  {
; CHECK: vmulps  %ymm1, %ymm0, %ymm0
entry:
  %0 = tail call <8 x float> asm "vmulps $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testymm_6(<8 x float> %_ymm0, <8 x float> %_ymm1)  {
; CHECK: vmulpd  %ymm1, %ymm0, %ymm0
entry:
  %0 = tail call <8 x float> asm "vmulpd $1, $2, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testymm_7(<8 x float> %_ymm0, <8 x float> %_ymm1)  {
; CHECK: vmovups %ymm1, %ymm0
entry:
  %0 = tail call <8 x float> asm "vmovups $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testymm_8(<8 x float> %_ymm0, <8 x float> %_ymm1)  {
; CHECK: vmovupd %ymm1, %ymm0
entry:
  %0 = tail call <8 x float> asm "vmovupd $1, $0", "=v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

