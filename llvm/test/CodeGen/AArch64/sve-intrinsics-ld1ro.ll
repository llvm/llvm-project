; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve,+f64mm -asm-verbose=0 < %s | FileCheck %s

;
; LD1ROB
;

define <vscale x 16 x i8> @ld1rob_i8(<vscale x 16 x i1> %pred, i8* %addr) nounwind {
; CHECK-LABEL: ld1rob_i8:
; CHECK-NEXT:  ld1rob { z0.b }, p0/z, [x0]
; CHECK-NEXT:  ret
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 16 x i8> %res
}

;
; LD1ROH
;

define <vscale x 8 x i16> @ld1roh_i16(<vscale x 8 x i1> %pred, i16* %addr) nounwind {
; CHECK-LABEL: ld1roh_i16:
; CHECK-NEXT:  ld1roh { z0.h }, p0/z, [x0]
; CHECK-NEXT:  ret
  %res = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1ro.nxv8i16(<vscale x 8 x i1> %pred, i16* %addr)
  ret <vscale x 8 x i16> %res
}

define <vscale x 8 x half> @ld1roh_half(<vscale x 8 x i1> %pred, half* %addr) nounwind {
; CHECK-LABEL: ld1roh_half:
; CHECK-NEXT:  ld1roh { z0.h }, p0/z, [x0]
; CHECK-NEXT:  ret
  %res = call <vscale x 8 x half> @llvm.aarch64.sve.ld1ro.nxv8f16(<vscale x 8 x i1> %pred, half* %addr)
  ret <vscale x 8 x half> %res
}

;
; LD1ROW
;

define <vscale x 4 x i32> @ld1row_i32(<vscale x 4 x i1> %pred, i32* %addr) nounwind {
; CHECK-LABEL: ld1row_i32:
; CHECK-NEXT:  ld1row { z0.s }, p0/z, [x0]
; CHECK-NEXT:  ret
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1ro.nxv4i32(<vscale x 4 x i1> %pred, i32* %addr)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x float> @ld1row_float(<vscale x 4 x i1> %pred, float* %addr) nounwind {
; CHECK-LABEL: ld1row_float:
; CHECK-NEXT:  ld1row { z0.s }, p0/z, [x0]
; CHECK-NEXT:  ret
  %res = call <vscale x 4 x float> @llvm.aarch64.sve.ld1ro.nxv4f32(<vscale x 4 x i1> %pred, float* %addr)
  ret <vscale x 4 x float> %res
}

;
; LD1ROD
;

define <vscale x 2 x i64> @ld1rod_i64(<vscale x 2 x i1> %pred, i64* %addr) nounwind {
; CHECK-LABEL: ld1rod_i64:
; CHECK-NEXT:  ld1rod { z0.d }, p0/z, [x0]
; CHECK-NEXT:  ret
  %res = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1ro.nxv2i64(<vscale x 2 x i1> %pred, i64* %addr)
  ret <vscale x 2 x i64> %res
}

define <vscale x 2 x double> @ld1rod_double(<vscale x 2 x i1> %pred, double* %addr) nounwind {
; CHECK-LABEL: ld1rod_double:
; CHECK-NEXT:  ld1rod { z0.d }, p0/z, [x0]
; CHECK-NEXT:  ret
  %res = call <vscale x 2 x double> @llvm.aarch64.sve.ld1ro.nxv2f64(<vscale x 2 x i1> %pred, double* %addr)
  ret <vscale x 2 x double> %res
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1>, i8*)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ld1ro.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ld1ro.nxv8f16(<vscale x 8 x i1>, half*)

declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1ro.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1ro.nxv4f32(<vscale x 4 x i1>, float*)

declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1ro.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1ro.nxv2f64(<vscale x 2 x i1>, double*)
