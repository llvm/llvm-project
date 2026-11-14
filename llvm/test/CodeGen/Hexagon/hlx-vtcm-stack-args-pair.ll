; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx,+hvxv85,+hvx-length128b -stop-after=hexagon-isel -verify-machineinstrs < %s | FileCheck %s

; Halfword and byte HLX pairs that do not fit in LL0-LL5 are passed on the
; VTCM stack as two 2048-byte halves.

; CHECK-LABEL: name: pair_h_args
; CHECK: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0)
; CHECK: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0 + 2048)
; CHECK: %{{[0-9]+}}:hlxllr = REG_SEQUENCE
define <2048 x i16> @pair_h_args(<2048 x i16> %a0, <2048 x i16> %a1, <2048 x i16> %a2, <2048 x i16> %a3, <2048 x i16> %a4, <2048 x i16> %a5, <2048 x i16> %a6) {
  ret <2048 x i16> %a6
}

; CHECK-LABEL: name: pair_h_caller
; CHECK-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack)
; CHECK-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 2048)
; CHECK: J2_call @pair_h_args
define <2048 x i16> @pair_h_caller(<2048 x i16> %x) {
  %r = call <2048 x i16> @pair_h_args(<2048 x i16> %x, <2048 x i16> %x, <2048 x i16> %x, <2048 x i16> %x, <2048 x i16> %x, <2048 x i16> %x, <2048 x i16> %x)
  ret <2048 x i16> %r
}

; CHECK-LABEL: name: pair_b_args
; CHECK: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0)
; CHECK: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0 + 2048)
; CHECK: %{{[0-9]+}}:hlxllr = REG_SEQUENCE
define <4096 x i8> @pair_b_args(<4096 x i8> %a0, <4096 x i8> %a1, <4096 x i8> %a2, <4096 x i8> %a3, <4096 x i8> %a4, <4096 x i8> %a5, <4096 x i8> %a6) {
  ret <4096 x i8> %a6
}

; CHECK-LABEL: name: pair_b_caller
; CHECK-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack)
; CHECK-DAG: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 2048)
; CHECK: J2_call @pair_b_args
define <4096 x i8> @pair_b_caller(<4096 x i8> %x) {
  %r = call <4096 x i8> @pair_b_args(<4096 x i8> %x, <4096 x i8> %x, <4096 x i8> %x, <4096 x i8> %x, <4096 x i8> %x, <4096 x i8> %x, <4096 x i8> %x)
  ret <4096 x i8> %r
}
