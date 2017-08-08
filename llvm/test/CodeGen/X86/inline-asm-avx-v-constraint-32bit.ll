; RUN: not llc < %s -mtriple i386-unknown-linux-gnu -mattr +avx -o /dev/null 2> %t
; RUN: FileCheck %s --input-file %t

define <4 x float> @testXMM_1(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <4 x float> asm "vmovhlps $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l, <4 x float> %_xmm0)
  ret <4 x float> %0
}

define <4 x float> @testXMM_2(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <4 x float> asm "movapd $1, $0", "=v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l)
  ret <4 x float> %0
}

define <4 x float> @testXMM_3(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <4 x float> asm "vmovapd $1, $0", "=v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l)
  ret <4 x float> %0
}

define <4 x float> @testXMM_4(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <4 x float> asm "vmpsadbw $$0, $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l, <4 x float> %_xmm0)
  ret <4 x float> %0
}

define <4 x float> @testXMM_5(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <4 x float> asm "vminpd $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l, i32 %_l)
  ret <4 x float> %0
}

define i32 @testXMM_6(i32 returned %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  tail call void asm sideeffect "vmovd $0, %eax", "v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l)
  ret i32 %_l
}

define <4 x float> @testXMM_7(<4 x float> returned %_xmm0) {
; CHECK: error: inline assembly requires more registers than available
entry:
  tail call void asm sideeffect "vmovmskps $0, %eax", "v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(<4 x float> %_xmm0)
  ret <4 x float> %_xmm0
}

define i32 @testXMM_8(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call i32 asm "vmulsd $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l, <4 x float> %_xmm0)
  ret i32 %0
}

define <4 x float> @testXMM_9(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <4 x float> asm "vorpd $1, $2, $0", "=v,v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l, <4 x float> %_xmm0)
  ret <4 x float> %0
}

define <4 x float> @testXMM_10(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <4 x float> asm "pabsb $1, $0", "=v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l)
  ret <4 x float> %0
}

define <4 x float> @testXMM_11(<4 x float> %_xmm0, i32 %_l) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <4 x float> asm "vpabsd $1, $0", "=v,v,~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"(i32 %_l)
  ret <4 x float> %0
}

define <8 x float> @testYMM_1(<8 x float> %_ymm0, <8 x float> %_ymm1) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <8 x float> asm "vmovsldup $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM_2(<8 x float> %_ymm0, <8 x float> %_ymm1) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <8 x float> asm "vmovapd $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testYMM_3(<8 x float> %_ymm0, <8 x float> %_ymm1) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <8 x float> asm "vminpd $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM_4(<8 x float> %_ymm0, <8 x float> %_ymm1) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <8 x float> asm "vorpd $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM(<8 x float> %_ymm0, <8 x float> %_ymm1) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <8 x float> asm "vmulps $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM_6(<8 x float> %_ymm0, <8 x float> %_ymm1) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <8 x float> asm "vmulpd $1, $2, $0", "=v,v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1, <8 x float> %_ymm0)
  ret <8 x float> %0
}

define <8 x float> @testYMM_7(<8 x float> %_ymm0, <8 x float> %_ymm1) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <8 x float> asm "vmovups $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

define <8 x float> @testYMM_8(<8 x float> %_ymm0, <8 x float> %_ymm1) {
; CHECK: error: inline assembly requires more registers than available
entry:
  %0 = tail call <8 x float> asm "vmovupd $1, $0", "=v,v,~{ymm0},~{ymm1},~{ymm2},~{ymm3},~{ymm4},~{ymm5},~{ymm6},~{ymm7},~{dirflag},~{fpsr},~{flags}"(<8 x float> %_ymm1)
  ret <8 x float> %0
}

