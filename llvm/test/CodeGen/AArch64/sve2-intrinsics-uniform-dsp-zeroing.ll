; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 -asm-verbose=0 -mattr=+use-experimental-zeroing-pseudos < %s | FileCheck %s

;
; SQSHLU
;

define <vscale x 16 x i8> @sqshlu_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: sqshlu_i8:
; CHECK:      movprfx z0.b, p0/z, z0.b
; CHECK-NEXT: sqshlu z0.b, p0/m, z0.b, #2
; CHECK-NEXT: ret
  %a_z = select <vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> zeroinitializer
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqshlu.nxv16i8(<vscale x 16 x i1> %pg,
                                                                  <vscale x 16 x i8> %a_z,
                                                                  i32 2)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqshlu_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: sqshlu_i16:
; CHECK:      movprfx z0.h, p0/z, z0.h
; CHECK-NEXT: sqshlu z0.h, p0/m, z0.h, #3
; CHECK-NEXT: ret
  %a_z = select <vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> zeroinitializer
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqshlu.nxv8i16(<vscale x 8 x i1> %pg,
                                                                  <vscale x 8 x i16> %a_z,
                                                                  i32 3)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqshlu_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: sqshlu_i32:
; CHECK:      movprfx z0.s, p0/z, z0.s
; CHECK-NEXT: sqshlu z0.s, p0/m, z0.s, #29
; CHECK-NEXT: ret
  %a_z = select <vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> zeroinitializer
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqshlu.nxv4i32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x i32> %a_z,
                                                                  i32 29)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @sqshlu_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: sqshlu_i64:
; CHECK:      movprfx z0.d, p0/z, z0.d
; CHECK-NEXT: sqshlu z0.d, p0/m, z0.d, #62
; CHECK-NEXT: ret
  %a_z = select <vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> zeroinitializer
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.sqshlu.nxv2i64(<vscale x 2 x i1> %pg,
                                                                  <vscale x 2 x i64> %a_z,
                                                                  i32 62)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqshlu.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqshlu.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqshlu.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, i32)
declare <vscale x 2 x i64> @llvm.aarch64.sve.sqshlu.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>, i32)
