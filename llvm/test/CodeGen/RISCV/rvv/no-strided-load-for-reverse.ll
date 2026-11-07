; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=GENERIC
; RUN: llc -mtriple=riscv64 -mcpu=spacemit-x100 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=SPACEMIT

; e8: spacemit should use vle+vrgather, generic should use vlse
define <vscale x 16 x i8> @reverse_load_e8(ptr %a, ptr %b, i32 zeroext %evl) {
; GENERIC-LABEL: reverse_load_e8:
; GENERIC:         vlse8.v
; GENERIC-NOT:     vrgather
;
; SPACEMIT-LABEL: reverse_load_e8:
; SPACEMIT-NOT:    vlse8.v
; SPACEMIT:        vle8.v
; SPACEMIT:        vrgatherei16.vv
entry:
  %load.a = call <vscale x 16 x i8> @llvm.vp.load.nxv16i8.p0(ptr %a, <vscale x 16 x i1> splat (i1 true), i32 %evl)
  %load.b = call <vscale x 16 x i8> @llvm.vp.load.nxv16i8.p0(ptr %b, <vscale x 16 x i1> splat (i1 true), i32 %evl)
  %rev.b = call <vscale x 16 x i8> @llvm.experimental.vp.reverse.nxv16i8(<vscale x 16 x i8> %load.b, <vscale x 16 x i1> splat (i1 true), i32 %evl)
  %add = add <vscale x 16 x i8> %rev.b, %load.a
  ret <vscale x 16 x i8> %add
}

; e16: spacemit should use vle+vrgather, generic should use vlse
define <vscale x 8 x i16> @reverse_load_e16(ptr %a, ptr %b, i32 zeroext %evl) {
; GENERIC-LABEL: reverse_load_e16:
; GENERIC:         vlse16.v
; GENERIC-NOT:     vrgather
;
; SPACEMIT-LABEL: reverse_load_e16:
; SPACEMIT-NOT:    vlse16.v
; SPACEMIT:        vle16.v
; SPACEMIT:        vrgather.vv
entry:
  %load.a = call <vscale x 8 x i16> @llvm.vp.load.nxv8i16.p0(ptr %a, <vscale x 8 x i1> splat (i1 true), i32 %evl)
  %load.b = call <vscale x 8 x i16> @llvm.vp.load.nxv8i16.p0(ptr %b, <vscale x 8 x i1> splat (i1 true), i32 %evl)
  %rev.b = call <vscale x 8 x i16> @llvm.experimental.vp.reverse.nxv8i16(<vscale x 8 x i16> %load.b, <vscale x 8 x i1> splat (i1 true), i32 %evl)
  %add = add <vscale x 8 x i16> %rev.b, %load.a
  ret <vscale x 8 x i16> %add
}

define <vscale x 4 x i32> @reverse_load_e32(ptr %a, ptr %b, i32 zeroext %evl) {
; GENERIC-LABEL: reverse_load_e32:
; GENERIC:         vlse32.v
;
; SPACEMIT-LABEL: reverse_load_e32:
; SPACEMIT:        vlse32.v
entry:
  %load.a = call <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p0(ptr %a, <vscale x 4 x i1> splat (i1 true), i32 %evl)
  %load.b = call <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p0(ptr %b, <vscale x 4 x i1> splat (i1 true), i32 %evl)
  %rev.b = call <vscale x 4 x i32> @llvm.experimental.vp.reverse.nxv4i32(<vscale x 4 x i32> %load.b, <vscale x 4 x i1> splat (i1 true), i32 %evl)
  %add = add <vscale x 4 x i32> %rev.b, %load.a
  ret <vscale x 4 x i32> %add
}

declare <vscale x 16 x i8> @llvm.vp.load.nxv16i8.p0(ptr, <vscale x 16 x i1>, i32)
declare <vscale x 16 x i8> @llvm.experimental.vp.reverse.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, i32)
declare <vscale x 8 x i16> @llvm.vp.load.nxv8i16.p0(ptr, <vscale x 8 x i1>, i32)
declare <vscale x 8 x i16> @llvm.experimental.vp.reverse.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, i32)
declare <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p0(ptr, <vscale x 4 x i1>, i32)
declare <vscale x 4 x i32> @llvm.experimental.vp.reverse.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32)
