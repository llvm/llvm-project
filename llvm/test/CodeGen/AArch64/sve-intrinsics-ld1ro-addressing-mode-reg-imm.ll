; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve,+f64mm -asm-verbose=0 < %s | FileCheck %s
;
; LD1ROB
;

define <vscale x 16 x i8> @ld1rob_i8(<vscale x 16 x i1> %pg, i8* %a) nounwind {
; CHECK-LABEL: ld1rob_i8:
; CHECK-NEXT:  ld1rob { z0.b }, p0/z, [x0, #32]
; CHECK-NEXT:  ret
  %base = getelementptr i8, i8* %a, i64 32
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1> %pg, i8* %base)
  ret <vscale x 16 x i8> %load
}

;
; LD1ROH
;

define <vscale x 8 x i16> @ld1roh_i16(<vscale x 8 x i1> %pg, i16* %a) nounwind {
; CHECK-LABEL: ld1roh_i16:
; CHECK-NEXT:  ld1roh { z0.h }, p0/z, [x0, #64]
; CHECK-NEXT:  ret
  %base = getelementptr i16, i16* %a, i64 32
  %load = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1ro.nxv8i16(<vscale x 8 x i1> %pg, i16* %base)
  ret <vscale x 8 x i16> %load
}

define <vscale x 8 x half> @ld1roh_f16(<vscale x 8 x i1> %pg, half* %a) nounwind {
; CHECK-LABEL: ld1roh_f16:
; CHECK-NEXT:  ld1roh { z0.h }, p0/z, [x0, #64]
; CHECK-NEXT:  ret
  %base = getelementptr half, half* %a, i64 32
  %load = call <vscale x 8 x half> @llvm.aarch64.sve.ld1ro.nxv8f16(<vscale x 8 x i1> %pg, half* %base)
  ret <vscale x 8 x half> %load
}

define <vscale x 8 x bfloat> @ld1roh_bf16(<vscale x 8 x i1> %pg, bfloat* %a) nounwind #0 {
; CHECK-LABEL: ld1roh_bf16:
; CHECK-NEXT:  ld1roh { z0.h }, p0/z, [x0, #64]
; CHECK-NEXT:  ret
  %base = getelementptr bfloat, bfloat* %a, i64 32
  %load = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1ro.nxv8bf16(<vscale x 8 x i1> %pg, bfloat* %base)
  ret <vscale x 8 x bfloat> %load
}

;
; LD1ROW
;

define<vscale x 4 x i32> @ld1row_i32(<vscale x 4 x i1> %pg, i32* %a) nounwind {
; CHECK-LABEL: ld1row_i32:
; CHECK-NEXT:  ld1row { z0.s }, p0/z, [x0, #128]
; CHECK-NEXT:  ret
  %base = getelementptr i32, i32* %a, i64 32
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1ro.nxv4i32(<vscale x 4 x i1> %pg, i32* %base)
  ret <vscale x 4 x i32> %load
}

define<vscale x 4 x float> @ld1row_f32(<vscale x 4 x i1> %pg, float* %a) nounwind {
; CHECK-LABEL: ld1row_f32:
; CHECK-NEXT:  ld1row { z0.s }, p0/z, [x0, #128]
; CHECK-NEXT:  ret
  %base = getelementptr float, float* %a, i64 32
  %load = call <vscale x 4 x float> @llvm.aarch64.sve.ld1ro.nxv4f32(<vscale x 4 x i1> %pg, float* %base)
  ret <vscale x 4 x float> %load
}

;
; LD1ROD
;

define <vscale x 2 x i64> @ld1rod_i64(<vscale x 2 x i1> %pg, i64* %a) nounwind {
; CHECK-LABEL: ld1rod_i64:
; CHECK-NEXT:  ld1rod { z0.d }, p0/z, [x0, #-64]
; CHECK-NEXT:  ret
  %base = getelementptr i64, i64* %a, i64 -8
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1ro.nxv2i64(<vscale x 2 x i1> %pg, i64* %base)
  ret <vscale x 2 x i64> %load
}

define <vscale x 2 x double> @ld1rod_f64(<vscale x 2 x i1> %pg, double* %a) nounwind {
; CHECK-LABEL: ld1rod_f64:
; CHECK-NEXT:  ld1rod { z0.d }, p0/z, [x0, #-128]
; CHECK-NEXT:  ret
  %base = getelementptr double, double* %a, i64 -16
  %load = call <vscale x 2 x double> @llvm.aarch64.sve.ld1ro.nxv2f64(<vscale x 2 x i1> %pg, double* %base)
  ret <vscale x 2 x double> %load
}


;;;;;;;;;;;;;;
; range checks: immediate must be a multiple of 32 in the range -256, ..., 224

; lower bound
define <vscale x 16 x i8> @ld1rob_i8_lower_bound(<vscale x 16 x i1> %pg, i8* %a) nounwind {
; CHECK-LABEL: ld1rob_i8_lower_bound:
; CHECK-NEXT:  ld1rob { z0.b }, p0/z, [x0, #-256]
; CHECK-NEXT:  ret
  %base = getelementptr i8, i8* %a, i64 -256
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1> %pg, i8* %base)
  ret <vscale x 16 x i8> %load
}

; below lower bound
define <vscale x 8 x i16> @ld1roh_i16_below_lower_bound(<vscale x 8 x i1> %pg, i16* %a) nounwind {
; CHECK-LABEL: ld1roh_i16_below_lower_bound:
; CHECK-NEXT:  mov     x[[IDX:[0-9]+]], #-129
; CHECK-NEXT:  ld1roh { z0.h }, p0/z, [x0, x[[IDX]], lsl #1]
; CHECK-NEXT:  ret
  %base = getelementptr i16, i16* %a, i64 -129
  %load = call <vscale x 8 x i16> @llvm.aarch64.sve.ld1ro.nxv8i16(<vscale x 8 x i1> %pg, i16* %base)
  ret <vscale x 8 x i16> %load
}

define <vscale x 16 x i8> @ld1rob_i8_below_lower_bound_01(<vscale x 16 x i1> %pg, i8* %a) nounwind {
; CHECK-LABEL: ld1rob_i8_below_lower_bound_01:
; CHECK-NEXT:  mov x[[OFFSET:[0-9]+]], #-257
; CHECK-NEXT:  ld1rob { z0.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT:  ret
  %base = getelementptr i8, i8* %a, i64 -257
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1> %pg, i8* %base)
  ret <vscale x 16 x i8> %load
}

; not a multiple of 32
define<vscale x 4 x i32> @ld1row_i32_not_multiple(<vscale x 4 x i1> %pg, i32* %a) nounwind {
; CHECK-LABEL: ld1row_i32_not_multiple:
; CHECK-NEXT:  mov x[[IDX:[0-9]+]], #3
; CHECK-NEXT:  ld1row { z0.s }, p0/z, [x0, x[[IDX]], lsl #2]
; CHECK-NEXT:  ret
  %base = getelementptr i32, i32* %a, i64 3
  %load = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1ro.nxv4i32(<vscale x 4 x i1> %pg, i32* %base)
  ret <vscale x 4 x i32> %load
}

; upper bound
define <vscale x 2 x i64> @ld1rod_i64_upper_bound(<vscale x 2 x i1> %pg, i64* %a) nounwind {
; CHECK-LABEL: ld1rod_i64_upper_bound:
; CHECK-NEXT:  ld1rod { z0.d }, p0/z, [x0, #224]
; CHECK-NEXT:  ret
  %base = getelementptr i64, i64* %a, i64 28
  %load = call <vscale x 2 x i64> @llvm.aarch64.sve.ld1ro.nxv2i64(<vscale x 2 x i1> %pg, i64* %base)
  ret <vscale x 2 x i64> %load
}

define <vscale x 16 x i8> @ld1rob_i8_beyond_upper_bound(<vscale x 16 x i1> %pg, i8* %a) nounwind {
; CHECK-LABEL: ld1rob_i8_beyond_upper_bound:
; CHECK-NEXT:  mov w[[OFFSET:[0-9]+]], #225
; CHECK-NEXT:  ld1rob { z0.b }, p0/z, [x0, x[[OFFSET]]]
; CHECK-NEXT:  ret
  %base = getelementptr i8, i8* %a, i64 225
  %load = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1> %pg, i8* %base)
  ret <vscale x 16 x i8> %load
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1ro.nxv16i8(<vscale x 16 x i1>, i8*)

declare <vscale x 8 x i16> @llvm.aarch64.sve.ld1ro.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ld1ro.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.ld1ro.nxv8bf16(<vscale x 8 x i1>, bfloat*)

declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1ro.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ld1ro.nxv4f32(<vscale x 4 x i1>, float*)

declare <vscale x 2 x i64> @llvm.aarch64.sve.ld1ro.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1ro.nxv2f64(<vscale x 2 x i1>, double*)


; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+f64mm,+bf16" }
