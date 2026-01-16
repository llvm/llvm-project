; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve -asm-verbose=0 < %s | FileCheck %s

;
; Masked Loads
;

define <vscale x 1 x i64> @masked_load_nxv1i64(ptr %a, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv1i64:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 1 x i64> @llvm.masked.load.nxv1i64(ptr %a, i32 8, <vscale x 1 x i1> %mask, <vscale x 1 x i64> poison)
  ret <vscale x 1 x i64> %load
}

define <vscale x 2 x i64> @masked_load_nxv2i64(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2i64:
; CHECK-NEXT: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64(ptr %a, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x i64> poison)
  ret <vscale x 2 x i64> %load
}

; TODO: This is sub-optimal, we should use an extending ld1w z.d.
define <vscale x 1 x i32> @masked_load_nxv1i32(ptr %a, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv1i32:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    uzp1 p1.d, p1.d, p1.d
; CHECK-NEXT:    uzp1 p0.s, p0.s, p1.s
; CHECK-NEXT:    ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 1 x i32> @llvm.masked.load.nxv1i32(ptr %a, i32 4, <vscale x 1 x i1> %mask, <vscale x 1 x i32> poison)
  ret <vscale x 1 x i32> %load
}

define <vscale x 4 x i32> @masked_load_nxv4i32(ptr %a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4i32:
; CHECK-NEXT: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32(ptr %a, i32 4, <vscale x 4 x i1> %mask, <vscale x 4 x i32> poison)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x i32> @masked_load_nxv4i32_i64_mask(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4i32_i64_mask:
; CHECK:         ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 4 x i1> @llvm.vector.stretch.nxv4i1.nxv2i1(<vscale x 2 x i1> %mask)
  %load = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32(ptr %a, i32 4, <vscale x 4 x i1> %mask.stretched, <vscale x 4 x i32> poison)
  ret <vscale x 4 x i32> %load
}

define <vscale x 4 x i32> @masked_load_nxv4i32_i64_mask_icmp(ptr %a, <vscale x 2 x i64> %v1.legalabi, <vscale x 2 x i64> %v2.legalabi) nounwind {
; CHECK-LABEL: masked_load_nxv4i32_i64_mask_icmp:
; CHECK:         ptrue p0.d
; CHECK-NEXT:    cmpeq p1.d, p0/z, z0.d, z1.d
; CHECK-NEXT:    zip1 p0.d, p1.d, p1.d
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v1 = call <vscale x 1 x i64> @llvm.vector.extract.nxv1i64.nxv2i64(<vscale x 2 x i64> %v1.legalabi, i64 0)
  %v2 = call <vscale x 1 x i64> @llvm.vector.extract.nxv1i64.nxv2i64(<vscale x 2 x i64> %v2.legalabi, i64 0)
  %mask = icmp eq <vscale x 1 x i64> %v1, %v2
  %mask.stretched = call <vscale x 4 x i1> @llvm.vector.stretch.nxv4i1.nxv2i1(<vscale x 1 x i1> %mask)
  %load = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32(ptr %a, i32 4, <vscale x 4 x i1> %mask.stretched, <vscale x 4 x i32> poison)
  ret <vscale x 4 x i32> %load
}

; TODO: This is sub-optimal, we should use an extending ld1h z.d.
define <vscale x 1 x i16> @masked_load_nxv1i16(ptr %a, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv1i16:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    uzp1 p1.d, p1.d, p1.d
; CHECK-NEXT:    uzp1 p0.s, p0.s, p1.s
; CHECK-NEXT:    uzp1 p1.s, p1.s, p1.s
; CHECK-NEXT:    uzp1 p0.h, p0.h, p1.h
; CHECK-NEXT:    ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 1 x i16> @llvm.masked.load.nxv1i16(ptr %a, i32 2, <vscale x 1 x i1> %mask, <vscale x 1 x i16> poison)
  ret <vscale x 1 x i16> %load
}

define <vscale x 8 x i16> @masked_load_nxv8i16(ptr %a, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8i16:
; CHECK-NEXT: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16(ptr %a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x i16> poison)
  ret <vscale x 8 x i16> %load
}

define <vscale x 8 x i16> @masked_load_nxv8i16_i64_mask(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8i16_i64_mask:
; CHECK:         ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv2i1(<vscale x 2 x i1> %mask)
  %load = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16(ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched, <vscale x 8 x i16> poison)
  ret <vscale x 8 x i16> %load
}

define <vscale x 8 x i16> @masked_load_nxv8i16_i64_mask_multi_stretch(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8i16_i64_mask_multi_stretch:
; CHECK:         ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 4 x i1> @llvm.vector.stretch.nxv4i1.nxv2i1(<vscale x 2 x i1> %mask)
  %mask.stretched2 = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv4i1(<vscale x 4 x i1> %mask.stretched)
  %load = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16(ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched2, <vscale x 8 x i16> poison)
  ret <vscale x 8 x i16> %load
}

define <vscale x 8 x i16> @masked_load_nxv8i16_i32_mask(ptr %a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8i16_i32_mask:
; CHECK:         ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv4i1(<vscale x 4 x i1> %mask)
  %load = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16(ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched, <vscale x 8 x i16> poison)
  ret <vscale x 8 x i16> %load
}

; TODO: This is sub-optimal, we should use an extending ld1b z.d.
define <vscale x 1 x i8> @masked_load_nxv1i8(ptr %a, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv1i8:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    uzp1 p1.d, p1.d, p1.d
; CHECK-NEXT:    uzp1 p0.s, p0.s, p1.s
; CHECK-NEXT:    uzp1 p1.s, p1.s, p1.s
; CHECK-NEXT:    uzp1 p0.h, p0.h, p1.h
; CHECK-NEXT:    uzp1 p1.h, p1.h, p1.h
; CHECK-NEXT:    uzp1 p0.b, p0.b, p1.b
; CHECK-NEXT:    ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 1 x i8> @llvm.masked.load.nxv1i8(ptr %a, i32 1, <vscale x 1 x i1> %mask, <vscale x 1 x i8> poison)
  ret <vscale x 1 x i8> %load
}

define <vscale x 16 x i8> @masked_load_nxv16i8(ptr %a, <vscale x 16 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv16i8:
; CHECK-NEXT: ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8(ptr %a, i32 1, <vscale x 16 x i1> %mask, <vscale x 16 x i8> poison)
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @masked_load_nxv16i8_i64_mask(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv16i8_i64_mask:
; CHECK:         ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 16 x i1> @llvm.vector.stretch.nxv16i1.nxv2i1(<vscale x 2 x i1> %mask)
  %load = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8(ptr %a, i32 1, <vscale x 16 x i1> %mask.stretched, <vscale x 16 x i8> poison)
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @masked_load_nxv16i8_i32_mask(ptr %a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv16i8_i32_mask:
; CHECK:         ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 16 x i1> @llvm.vector.stretch.nxv16i1.nxv4i1(<vscale x 4 x i1> %mask)
  %load = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8(ptr %a, i32 1, <vscale x 16 x i1> %mask.stretched, <vscale x 16 x i8> poison)
  ret <vscale x 16 x i8> %load
}

define <vscale x 16 x i8> @masked_load_nxv16i8_i16_mask(ptr %a, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv16i8_i16_mask:
; CHECK:         ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 16 x i1> @llvm.vector.stretch.nxv16i1.nxv8i1(<vscale x 8 x i1> %mask)
  %load = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8(ptr %a, i32 1, <vscale x 16 x i1> %mask.stretched, <vscale x 16 x i8> poison)
  ret <vscale x 16 x i8> %load
}

define <vscale x 1 x double> @masked_load_nxv1f64(ptr %a, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv1f64:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 1 x double> @llvm.masked.load.nxv1f64(ptr %a, i32 8, <vscale x 1 x i1> %mask, <vscale x 1 x double> poison)
  ret <vscale x 1 x double> %load
}

define <vscale x 2 x double> @masked_load_nxv2f64(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f64:
; CHECK-NEXT: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.masked.load.nxv2f64(ptr %a, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x double> poison)
  ret <vscale x 2 x double> %load
}

define <vscale x 1 x float> @masked_load_nxv1f32(ptr %a, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv1f32:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    ld1w { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 1 x float> @llvm.masked.load.nxv1f32(ptr %a, i32 4, <vscale x 1 x i1> %mask, <vscale x 1 x float> poison)
  ret <vscale x 1 x float> %load
}

define <vscale x 2 x float> @masked_load_nxv2f32(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f32:
; CHECK-NEXT: ld1w { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x float> @llvm.masked.load.nxv2f32(ptr %a, i32 4, <vscale x 2 x i1> %mask, <vscale x 2 x float> poison)
  ret <vscale x 2 x float> %load
}

define <vscale x 1 x half> @masked_load_nxv1f16(ptr %a, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv1f16:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 1 x half> @llvm.masked.load.nxv1f16(ptr %a, i32 2, <vscale x 1 x i1> %mask, <vscale x 1 x half> poison)
  ret <vscale x 1 x half> %load
}

define <vscale x 2 x half> @masked_load_nxv2f16(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f16:
; CHECK-NEXT: ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x half> @llvm.masked.load.nxv2f16(ptr %a, i32 2, <vscale x 2 x i1> %mask, <vscale x 2 x half> poison)
  ret <vscale x 2 x half> %load
}

define <vscale x 1 x bfloat> @masked_load_nxv1bf16(ptr %a, <vscale x 1 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_load_nxv1bf16:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %load = call <vscale x 1 x bfloat> @llvm.masked.load.nxv1bf16(ptr %a, i32 2, <vscale x 1 x i1> %mask, <vscale x 1 x bfloat> poison)
  ret <vscale x 1 x bfloat> %load
}

define <vscale x 2 x bfloat> @masked_load_nxv2bf16(ptr %a, <vscale x 2 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_load_nxv2bf16:
; CHECK-NEXT: ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x bfloat> @llvm.masked.load.nxv2bf16(ptr %a, i32 2, <vscale x 2 x i1> %mask, <vscale x 2 x bfloat> poison)
  ret <vscale x 2 x bfloat> %load
}

define <vscale x 4 x float> @masked_load_nxv4f32(ptr %a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4f32:
; CHECK-NEXT: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x float> @llvm.masked.load.nxv4f32(ptr %a, i32 4, <vscale x 4 x i1> %mask, <vscale x 4 x float> poison)
  ret <vscale x 4 x float> %load
}

define <vscale x 4 x half> @masked_load_nxv4f16(ptr %a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4f16:
; CHECK-NEXT: ld1h { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x half> @llvm.masked.load.nxv4f16(ptr %a, i32 2, <vscale x 4 x i1> %mask, <vscale x 4 x half> poison)
  ret <vscale x 4 x half> %load
}

define <vscale x 4 x bfloat> @masked_load_nxv4bf16(ptr %a, <vscale x 4 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_load_nxv4bf16:
; CHECK-NEXT: ld1h { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x bfloat> @llvm.masked.load.nxv4bf16(ptr %a, i32 2, <vscale x 4 x i1> %mask, <vscale x 4 x bfloat> poison)
  ret <vscale x 4 x bfloat> %load
}

define <vscale x 8 x half> @masked_load_nxv8f16(ptr %a, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8f16:
; CHECK-NEXT: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x half> @llvm.masked.load.nxv8f16(ptr %a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x half> poison)
  ret <vscale x 8 x half> %load
}

define <vscale x 8 x bfloat> @masked_load_nxv8bf16(ptr %a, <vscale x 8 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_load_nxv8bf16:
; CHECK-NEXT: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x bfloat> @llvm.masked.load.nxv8bf16(ptr %a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> poison)
  ret <vscale x 8 x bfloat> %load
}

define <vscale x 4 x i32> @masked_load_passthru(ptr %a, <vscale x 4 x i1> %mask, <vscale x 4 x i32> %passthru) nounwind {
; CHECK-LABEL: masked_load_passthru:
; CHECK-NEXT: ld1w { z1.s }, p0/z, [x0]
; CHECK-NEXT: mov z0.s, p0/m, z1.s
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32(ptr %a, i32 4, <vscale x 4 x i1> %mask, <vscale x 4 x i32> %passthru)
  ret <vscale x 4 x i32> %load
}

; Masked load requires promotion
define <vscale x 2 x i16> @masked_load_nxv2i16(ptr noalias %in, <vscale x 2 x i1> %mask) {
; CHECK-LABEL: masked_load_nxv2i16
; CHECK:       ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT:  ret
  %wide.load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(ptr %in, i32 2, <vscale x 2 x i1> %mask, <vscale x 2 x i16> poison)
  ret <vscale x 2 x i16> %wide.load
}

define <vscale x 4 x i16> @masked_load_nxv4i16_i64_mask(ptr noalias %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4i16_i64_mask:
; CHECK:         trn1 p0.s, p0.s, p0.s
; CHECK-NEXT:    ld1h { z0.s }, p0/z, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 4 x i1> @llvm.vector.stretch.nxv4i1.nxv2i1(<vscale x 2 x i1> %mask)
  %load = call <vscale x 4 x i16> @llvm.masked.load.nxv4i16(ptr %a, i32 2, <vscale x 4 x i1> %mask.stretched, <vscale x 4 x i16> poison)
  ret <vscale x 4 x i16> %load
}

;
; Masked Stores
;

define void @masked_store_nxv1i64(ptr %a, <vscale x 1 x i64> %val, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv1i64:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1i64(<vscale x 1 x i64> %val, ptr %a, i32 8, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_store_nxv2i64(ptr %a, <vscale x 2 x i64> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2i64:
; CHECK-NEXT: st1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2i64(<vscale x 2 x i64> %val, ptr %a, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

; TODO: This is sub-optimal, we should use a truncating store from d to s
define void @masked_store_nxv1i32(ptr %a, <vscale x 1 x i32> %val, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv1i32:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    uzp1 p1.d, p1.d, p1.d
; CHECK-NEXT:    uzp1 p0.s, p0.s, p1.s
; CHECK-NEXT:    st1w { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1i32(<vscale x 1 x i32> %val, ptr %a, i32 4, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_store_nxv2i32(ptr %a, <vscale x 2 x i32> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2i32:
; CHECK:         st1w { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv2i32(<vscale x 2 x i32> %val, ptr %a, i32 4, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv4i32(ptr %a, <vscale x 4 x i32> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4i32:
; CHECK-NEXT: st1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv4i32(<vscale x 4 x i32> %val, ptr %a, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_store_nxv4i32_i64_mask(ptr %a, <vscale x 4 x i32> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4i32_i64_mask:
; CHECK:         st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 4 x i1> @llvm.vector.stretch.nxv4i1.nxv2i1(<vscale x 2 x i1> %mask)
  call void @llvm.masked.store.nxv4i32(<vscale x 4 x i32> %val, ptr %a, i32 4, <vscale x 4 x i1> %mask.stretched)
  ret void
}

; TODO: This is sub-optimal, we should use a truncating store from d to h
define void @masked_store_nxv1i16(ptr %a, <vscale x 1 x i16> %val, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv1i16:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    uzp1 p1.d, p1.d, p1.d
; CHECK-NEXT:    uzp1 p0.s, p0.s, p1.s
; CHECK-NEXT:    uzp1 p1.s, p1.s, p1.s
; CHECK-NEXT:    uzp1 p0.h, p0.h, p1.h
; CHECK-NEXT:    st1h { z0.h }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1i16(<vscale x 1 x i16> %val, ptr %a, i32 2, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_store_nxv8i16(ptr %a, <vscale x 8 x i16> %val, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8i16:
; CHECK-NEXT: st1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv8i16(<vscale x 8 x i16> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask)
  ret void
}

define void @masked_store_nxv8i16_i64_mask(ptr %a, <vscale x 8 x i16> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8i16_i64_mask:
; CHECK:         st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv2i1(<vscale x 2 x i1> %mask)
  call void @llvm.masked.store.nxv8i16(<vscale x 8 x i16> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched)
  ret void
}

define void @masked_store_nxv8i16_i32_mask(ptr %a, <vscale x 8 x i16> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8i16_i32_mask:
; CHECK:         st1w { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv4i1(<vscale x 4 x i1> %mask)
  call void @llvm.masked.store.nxv8i16(<vscale x 8 x i16> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched)
  ret void
}

; TODO: This is sub-optimal, we should use a truncating store from d to b
define void @masked_store_nxv1i8(ptr %a, <vscale x 1 x i8> %val, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv1i8:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    uzp1 p1.d, p1.d, p1.d
; CHECK-NEXT:    uzp1 p0.s, p0.s, p1.s
; CHECK-NEXT:    uzp1 p1.s, p1.s, p1.s
; CHECK-NEXT:    uzp1 p0.h, p0.h, p1.h
; CHECK-NEXT:    uzp1 p1.h, p1.h, p1.h
; CHECK-NEXT:    uzp1 p0.b, p0.b, p1.b
; CHECK-NEXT:    st1b { z0.b }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1i8(<vscale x 1 x i8> %val, ptr %a, i32 1, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_store_nxv16i8(ptr %a, <vscale x 16 x i8> %val, <vscale x 16 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv16i8:
; CHECK-NEXT: st1b { z0.b }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv16i8(<vscale x 16 x i8> %val, ptr %a, i32 1, <vscale x 16 x i1> %mask)
  ret void
}

define void @masked_store_nxv16i8_i64_mask(ptr %a, <vscale x 16 x i8> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv16i8_i64_mask:
; CHECK:         st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 16 x i1> @llvm.vector.stretch.nxv16i1.nxv2i1(<vscale x 2 x i1> %mask)
  call void @llvm.masked.store.nxv16i8(<vscale x 16 x i8> %val, ptr %a, i32 1, <vscale x 16 x i1> %mask.stretched)
  ret void
}

define void @masked_store_nxv16i8_i32_mask(ptr %a, <vscale x 16 x i8> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv16i8_i32_mask:
; CHECK:         st1w { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 16 x i1> @llvm.vector.stretch.nxv16i1.nxv4i1(<vscale x 4 x i1> %mask)
  call void @llvm.masked.store.nxv16i8(<vscale x 16 x i8> %val, ptr %a, i32 1, <vscale x 16 x i1> %mask.stretched)
  ret void
}

define void @masked_store_nxv16i8_i16_mask(ptr %a, <vscale x 16 x i8> %val, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv16i8_i16_mask:
; CHECK:         st1h { z0.h }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 16 x i1> @llvm.vector.stretch.nxv16i1.nxv8i1(<vscale x 8 x i1> %mask)
  call void @llvm.masked.store.nxv16i8(<vscale x 16 x i8> %val, ptr %a, i32 1, <vscale x 16 x i1> %mask.stretched)
  ret void
}

define void @masked_store_nxv1f64(ptr %a, <vscale x 1 x double> %val, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv1f64:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1f64(<vscale x 1 x double> %val, ptr %a, i32 8, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_store_nxv2f64(ptr %a, <vscale x 2 x double> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2f64:
; CHECK-NEXT: st1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2f64(<vscale x 2 x double> %val, ptr %a, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv1f32(ptr %a, <vscale x 1 x float> %val, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv1f32:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    st1w { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1f32(<vscale x 1 x float> %val, ptr %a, i32 4, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_store_nxv2f32(ptr %a, <vscale x 2 x float> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2f32:
; CHECK-NEXT: st1w { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2f32(<vscale x 2 x float> %val, ptr %a, i32 4, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv1f16(ptr %a, <vscale x 1 x half> %val, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv1f16:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    st1h { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1f16(<vscale x 1 x half> %val, ptr %a, i32 4, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_store_nxv2f16(ptr %a, <vscale x 2 x half> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2f16:
; CHECK-NEXT: st1h { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2f16(<vscale x 2 x half> %val, ptr %a, i32 4, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv4f32(ptr %a, <vscale x 4 x float> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4f32:
; CHECK-NEXT: st1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv4f32(<vscale x 4 x float> %val, ptr %a, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_store_nxv4f32_i64_mask(ptr %a, <vscale x 4 x float> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4f32_i64_mask:
; CHECK:         st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 4 x i1> @llvm.vector.stretch.nxv4i1.nxv2i1(<vscale x 2 x i1> %mask)
  call void @llvm.masked.store.nxv4f32(<vscale x 4 x float> %val, ptr %a, i32 4, <vscale x 4 x i1> %mask.stretched)
  ret void
}

define void @masked_store_nxv4f16(ptr %a, <vscale x 4 x half> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4f16:
; CHECK-NEXT: st1h { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv4f16(<vscale x 4 x half> %val, ptr %a, i32 2, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_store_nxv8f16(ptr %a, <vscale x 8 x half> %val, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8f16:
; CHECK-NEXT: st1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv8f16(<vscale x 8 x half> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask)
  ret void
}

define void @masked_store_nxv8f16_i64_mask(ptr %a, <vscale x 8 x half> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8f16_i64_mask:
; CHECK:         st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv2i1(<vscale x 2 x i1> %mask)
  call void @llvm.masked.store.nxv8f16(<vscale x 8 x half> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched)
  ret void
}

define void @masked_store_nxv8f16_i32_mask(ptr %a, <vscale x 8 x half> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8f16_i32_mask:
; CHECK:         st1w { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv4i1(<vscale x 4 x i1> %mask)
  call void @llvm.masked.store.nxv8f16(<vscale x 8 x half> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched)
  ret void
}

define void @masked_store_nxv1bf16(ptr %a, <vscale x 1 x bfloat> %val, <vscale x 1 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_store_nxv1bf16:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    st1h { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1bf16(<vscale x 1 x bfloat> %val, ptr %a, i32 2, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_store_nxv2bf16(ptr %a, <vscale x 2 x bfloat> %val, <vscale x 2 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_store_nxv2bf16:
; CHECK-NEXT: st1h { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2bf16(<vscale x 2 x bfloat> %val, ptr %a, i32 2, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv4bf16(ptr %a, <vscale x 4 x bfloat> %val, <vscale x 4 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_store_nxv4bf16:
; CHECK-NEXT: st1h { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv4bf16(<vscale x 4 x bfloat> %val, ptr %a, i32 2, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_store_nxv8bf16(ptr %a, <vscale x 8 x bfloat> %val, <vscale x 8 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_store_nxv8bf16:
; CHECK-NEXT: st1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv8bf16(<vscale x 8 x bfloat> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask)
  ret void
}

define void @masked_store_nxv8bf16_i64_mask(ptr %a, <vscale x 8 x bfloat> %val, <vscale x 2 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_store_nxv8bf16_i64_mask:
; CHECK:         st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv2i1(<vscale x 2 x i1> %mask)
  call void @llvm.masked.store.nxv8bf16(<vscale x 8 x bfloat> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched)
  ret void
}

define void @masked_store_nxv8bf16_i32_mask(ptr %a, <vscale x 8 x bfloat> %val, <vscale x 4 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_store_nxv8bf16_i32_mask:
; CHECK:         st1w { z0.s }, p0, [x0]
; CHECK-NEXT:    ret
  %mask.stretched = call <vscale x 8 x i1> @llvm.vector.stretch.nxv8i1.nxv4i1(<vscale x 4 x i1> %mask)
  call void @llvm.masked.store.nxv8bf16(<vscale x 8 x bfloat> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask.stretched)
  ret void
}

;
; Masked load store of pointer data type
;

; Pointer of integer type

define <vscale x 1 x ptr> @masked.load.nxv1p0(ptr %vector_ptr, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv1p0:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 1 x ptr> @llvm.masked.load.nxv1p0(ptr %vector_ptr, i32 8, <vscale x 1 x i1> %mask, <vscale x 1 x ptr> poison)
  ret <vscale x 1 x ptr> %v
}

define <vscale x 2 x ptr> @masked.load.nxv2p0i8(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0i8:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> poison)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0i16(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0i16:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> poison)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0i32(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0i32:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> poison)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0i64(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0i64:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> poison)
  ret <vscale x 2 x ptr> %v
}

; Pointer of floating-point type

define <vscale x 2 x ptr> @masked.load.nxv2p0bf16(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked.load.nxv2p0bf16:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> poison)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0f16(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0f16:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> poison)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0f32(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0f32:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> poison)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0f64(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0f64:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> poison)
  ret <vscale x 2 x ptr> %v
}

; Pointer of array type

define void @masked.store.nxv1p0(<vscale x 1 x ptr> %data, ptr %vector_ptr, <vscale x 1 x i1> %mask) nounwind {
; CHECK-LABEL: masked.store.nxv1p0:
; CHECK:         pfalse p1.b
; CHECK-NEXT:    uzp1 p0.d, p0.d, p1.d
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv1p0(<vscale x 1 x ptr> %data, ptr %vector_ptr, i32 8, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked.store.nxv2p0a64i16(<vscale x 2 x ptr> %data, ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.store.nxv2p0a64i16:
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv2p0.p0(<vscale x 2 x ptr> %data, ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

; Pointer of struct type

%struct = type { ptr, i32 }
define void @masked.store.nxv2p0s_struct(<vscale x 2 x ptr> %data, ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.store.nxv2p0s_struct:
; CHECK-NEXT:    st1d { z0.d }, p0, [x0]
; CHECK-NEXT:    ret
  call void @llvm.masked.store.nxv2p0.p0(<vscale x 2 x ptr> %data, ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask)
  ret void
}


declare <vscale x 2 x i64> @llvm.masked.load.nxv2i64(ptr, i32, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32(ptr, i32, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 2 x i16> @llvm.masked.load.nxv2i16(ptr, i32, <vscale x 2 x i1>, <vscale x 2 x i16>)
declare <vscale x 8 x i16> @llvm.masked.load.nxv8i16(ptr, i32, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.masked.load.nxv16i8(ptr, i32, <vscale x 16 x i1>, <vscale x 16 x i8>)

declare <vscale x 2 x double> @llvm.masked.load.nxv2f64(ptr, i32, <vscale x 2 x i1>, <vscale x 2 x double>)
declare <vscale x 2 x float> @llvm.masked.load.nxv2f32(ptr, i32, <vscale x 2 x i1>, <vscale x 2 x float>)
declare <vscale x 2 x half> @llvm.masked.load.nxv2f16(ptr, i32, <vscale x 2 x i1>, <vscale x 2 x half>)
declare <vscale x 4 x float> @llvm.masked.load.nxv4f32(ptr, i32, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 4 x half> @llvm.masked.load.nxv4f16(ptr, i32, <vscale x 4 x i1>, <vscale x 4 x half>)
declare <vscale x 8 x half> @llvm.masked.load.nxv8f16(ptr, i32, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 2 x bfloat> @llvm.masked.load.nxv2bf16(ptr, i32, <vscale x 2 x i1>, <vscale x 2 x bfloat>)
declare <vscale x 4 x bfloat> @llvm.masked.load.nxv4bf16(ptr, i32, <vscale x 4 x i1>, <vscale x 4 x bfloat>)
declare <vscale x 8 x bfloat> @llvm.masked.load.nxv8bf16(ptr, i32, <vscale x 8 x i1>, <vscale x 8 x bfloat>)

declare void @llvm.masked.store.nxv2i64(<vscale x 2 x i64>, ptr, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv4i32(<vscale x 4 x i32>, ptr, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv8i16(<vscale x 8 x i16>, ptr, i32, <vscale x 8 x i1>)
declare void @llvm.masked.store.nxv16i8(<vscale x 16 x i8>, ptr, i32, <vscale x 16 x i1>)

declare void @llvm.masked.store.nxv2f64(<vscale x 2 x double>, ptr, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2f32(<vscale x 2 x float>, ptr, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2f16(<vscale x 2 x half>, ptr, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv4f32(<vscale x 4 x float>, ptr, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv4f16(<vscale x 4 x half>, ptr, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv8f16(<vscale x 8 x half>, ptr, i32, <vscale x 8 x i1>)
declare void @llvm.masked.store.nxv2bf16(<vscale x 2 x bfloat>, ptr, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv4bf16(<vscale x 4 x bfloat>, ptr, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv8bf16(<vscale x 8 x bfloat>, ptr, i32, <vscale x 8 x i1>)

declare <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr, i32 immarg, <vscale x 2 x i1>, <vscale x 2 x ptr>)


declare void @llvm.masked.store.nxv2p0.p0(<vscale x 2 x ptr>, ptr, i32 immarg, <vscale x 2 x i1>)


; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
