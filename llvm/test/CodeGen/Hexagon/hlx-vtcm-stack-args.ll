; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx,+hvxv85,+hvx-length128b -stop-after=hexagon-isel -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx,+hvxv85,+hvx-length128b -stop-after=prolog-epilog -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-PEI
; RUN: llc -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv85 -mattr=+hlx < %s | FileCheck %s --check-prefix=MUSL

; HLX arguments that do not fit in L0-L11 go on the VTCM stack. Arguments of
; other types after them keep their registers.
; CHECK-LABEL: name: hlx_hvx_args
; CHECK: liveins: $l0, $v0, $r0
; CHECK-DAG: %[[PTR:[0-9]+]]:intregs = COPY $r0
; CHECK-DAG: %[[HVX:[0-9]+]]:hvxvr = COPY $v0
; CHECK-DAG: %[[A0:[0-9]+]]:hlxlr = COPY $l0
; CHECK: %[[FI:[0-9]+]]:intregs = PS_fi %fixed-stack.0, 0
; CHECK: %[[A12:[0-9]+]]:hlxlr = LX_L16b_pseudo %[[FI]] :: (load (s16384) from %fixed-stack.0)
; CHECK: %[[SUM:[0-9]+]]:hlxlr = LX_add_h %[[A0]], %[[A12]]
; CHECK: LX_S16b_pseudo %[[PTR]], %[[SUM]]
; CHECK: $v0 = COPY %[[HVX]]
define <32 x i32> @hlx_hvx_args(<1024 x i16> %a0, <1024 x i16> %a1, <1024 x i16> %a2, <1024 x i16> %a3, <1024 x i16> %a4, <1024 x i16> %a5, <1024 x i16> %a6, <1024 x i16> %a7, <1024 x i16> %a8, <1024 x i16> %a9, <1024 x i16> %a10, <1024 x i16> %a11, <1024 x i16> %a12, <32 x i32> %h, ptr %p) {
  %s = add <1024 x i16> %a0, %a12
  store <1024 x i16> %s, ptr %p, align 2048
  ret <32 x i32> %h
}

; CHECK-LABEL: name: hlx_hvx_args_caller
; CHECK: %[[VSP:[0-9]+]]:intregs = COPY $vsp
; CHECK: LX_S16b_pseudo %[[VSP]], %{{[0-9]+}} :: (store (s16384) into stack)
; CHECK-NOT: V6_vS32b
; CHECK: J2_call @hlx_hvx_args, hexagoncsr_hlx
; CHECK-SAME: implicit $l11, implicit $v0, implicit $r0,
define <32 x i32> @hlx_hvx_args_caller(<1024 x i16> %x, <32 x i32> %h, ptr %p) {
  %r = call <32 x i32> @hlx_hvx_args(<1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <32 x i32> %h, ptr %p)
  ret <32 x i32> %r
}

; A function with its own VTCM frame finds its stack arguments just above it,
; at VSP + frame size.
; CHECK-PEI-LABEL: name: hlx_frame_and_args
; CHECK-PEI: frame-setup V2_vallocframe_imm 1
; CHECK-PEI-DAG: A2_addi killed $r{{[0-9]+}}, 2048
; CHECK-PEI-DAG: A2_addi killed $r{{[0-9]+}}, 4096
; CHECK-PEI-DAG: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.1)
; CHECK-PEI-DAG: LX_L16b_pseudo {{.*}} :: (load (s16384) from %fixed-stack.0)
; CHECK-PEI: V2_vdeallocframe_imm 1
define <1024 x i16> @hlx_frame_and_args(<1024 x i16> %a0, <1024 x i16> %a1, <1024 x i16> %a2, <1024 x i16> %a3, <1024 x i16> %a4, <1024 x i16> %a5, <1024 x i16> %a6, <1024 x i16> %a7, <1024 x i16> %a8, <1024 x i16> %a9, <1024 x i16> %a10, <1024 x i16> %a11, <1024 x i16> %a12, <1024 x i16> %a13) {
  %l = alloca <1024 x i16>, align 2048
  store volatile <1024 x i16> %a12, ptr %l, align 2048
  %v = load volatile <1024 x i16>, ptr %l, align 2048
  %r = add <1024 x i16> %v, %a13
  ret <1024 x i16> %r
}

; The outgoing argument area is at the bottom of the VTCM frame, below the
; local (vsp+6144) and the L12 save slot.
; CHECK-PEI-LABEL: name: hlx_local_and_outgoing
; CHECK-PEI: frame-setup V2_vallocframe_imm 4
; CHECK-PEI: A2_addi killed $r{{[0-9]+}}, 6144
; CHECK-PEI: LX_S16b_pseudo {{.*}} :: (volatile store (s16384) into %ir.l)
; CHECK-PEI: $[[VSP:r[0-9]+]] = COPY $vsp
; CHECK-PEI: LX_S16b_pseudo renamable $[[VSP]], {{.*}} :: (store (s16384) into stack)
; CHECK-PEI: LX_S16b_pseudo {{.*}} :: (store (s16384) into stack + 2048)
; CHECK-PEI: J2_call @hlx_frame_and_args
; CHECK-PEI: V2_vdeallocframe_imm 4
define <1024 x i16> @hlx_local_and_outgoing(<1024 x i16> %x) {
  %l = alloca <1024 x i16>, align 2048
  store volatile <1024 x i16> %x, ptr %l, align 2048
  %v = load volatile <1024 x i16>, ptr %l, align 2048
  %r = call <1024 x i16> @hlx_frame_and_args(<1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %x, <1024 x i16> %v, <1024 x i16> %v)
  ret <1024 x i16> %r
}

; With musl varargs, the prologue copies the named scalar stack arguments; HLX
; stack arguments are on the VTCM stack and are not copied.
; MUSL-LABEL: hlx_vararg:
; MUSL-NOT: memd(r29
; MUSL: r[[VSP:[0-9]+]] = vsp
; MUSL-NOT: memd(r29
; MUSL: L{{[0-9]+}}.tmp = mem(r[[VSP]])
; MUSL-NOT: memd(r29
; MUSL: jumpr r31
define <1024 x i16> @hlx_vararg(<1024 x i16> %a0, <1024 x i16> %a1, <1024 x i16> %a2, <1024 x i16> %a3, <1024 x i16> %a4, <1024 x i16> %a5, <1024 x i16> %a6, <1024 x i16> %a7, <1024 x i16> %a8, <1024 x i16> %a9, <1024 x i16> %a10, <1024 x i16> %a11, <1024 x i16> %a12, i32 %n, ...) {
  %r = add <1024 x i16> %a0, %a12
  ret <1024 x i16> %r
}
