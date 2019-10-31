; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve -asm-verbose=0 < %s | FileCheck %s

;
; Masked Loads
;

define <vscale x 2 x i64> @masked_load_nxv2i64(<vscale x 2 x i64> *%a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2i64:
; CHECK-NEXT: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64(<vscale x 2 x i64> *%a, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x i64> undef)
  ret <vscale x 2 x i64> %load
}

define <vscale x 4 x i32> @masked_load_nxv4i32(<vscale x 4 x i32> *%a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4i32:
; CHECK-NEXT: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32(<vscale x 4 x i32> *%a, i32 4, <vscale x 4 x i1> %mask, <vscale x 4 x i32> undef)
  ret <vscale x 4 x i32> %load
}

define <vscale x 8 x i16> @masked_load_nxv8i16(<vscale x 8 x i16> *%a, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8i16:
; CHECK-NEXT: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16(<vscale x 8 x i16> *%a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x i16> undef)
  ret <vscale x 8 x i16> %load
}

define <vscale x 16 x i8> @masked_load_nxv16i8(<vscale x 16 x i8> *%a, <vscale x 16 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv16i8:
; CHECK-NEXT: ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8(<vscale x 16 x i8> *%a, i32 1, <vscale x 16 x i1> %mask, <vscale x 16 x i8> undef)
  ret <vscale x 16 x i8> %load
}

define <vscale x 2 x double> @masked_load_nxv2f64(<vscale x 2 x double> *%a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f64:
; CHECK-NEXT: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.masked.load.nxv2f64(<vscale x 2 x double> *%a, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x double> undef)
  ret <vscale x 2 x double> %load
}

define <vscale x 2 x float> @masked_load_nxv2f32(<vscale x 2 x float> *%a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f32:
; CHECK-NEXT: ld1w { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x float> @llvm.masked.load.nxv2f32(<vscale x 2 x float> *%a, i32 4, <vscale x 2 x i1> %mask, <vscale x 2 x float> undef)
  ret <vscale x 2 x float> %load
}

define <vscale x 2 x half> @masked_load_nxv2f16(<vscale x 2 x half> *%a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f16:
; CHECK-NEXT: ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x half> @llvm.masked.load.nxv2f16(<vscale x 2 x half> *%a, i32 2, <vscale x 2 x i1> %mask, <vscale x 2 x half> undef)
  ret <vscale x 2 x half> %load
}

define <vscale x 4 x float> @masked_load_nxv4f32(<vscale x 4 x float> *%a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4f32:
; CHECK-NEXT: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x float> @llvm.masked.load.nxv4f32(<vscale x 4 x float> *%a, i32 4, <vscale x 4 x i1> %mask, <vscale x 4 x float> undef)
  ret <vscale x 4 x float> %load
}

define <vscale x 4 x half> @masked_load_nxv4f16(<vscale x 4 x half> *%a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4f16:
; CHECK-NEXT: ld1h { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x half> @llvm.masked.load.nxv4f16(<vscale x 4 x half> *%a, i32 2, <vscale x 4 x i1> %mask, <vscale x 4 x half> undef)
  ret <vscale x 4 x half> %load
}

define <vscale x 8 x half> @masked_load_nxv8f16(<vscale x 8 x half> *%a, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8f16:
; CHECK-NEXT: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x half> @llvm.masked.load.nxv8f16(<vscale x 8 x half> *%a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x half> undef)
  ret <vscale x 8 x half> %load
}

;
; Masked Stores
;

define void @masked_store_nxv2i64(<vscale x 2 x i64> *%a, <vscale x 2 x i64> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2i64:
; CHECK-NEXT: st1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2i64(<vscale x 2 x i64> %val, <vscale x 2 x i64> *%a, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv4i32(<vscale x 4 x i32> *%a, <vscale x 4 x i32> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4i32:
; CHECK-NEXT: st1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv4i32(<vscale x 4 x i32> %val, <vscale x 4 x i32> *%a, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_store_nxv8i16(<vscale x 8 x i16> *%a, <vscale x 8 x i16> %val, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8i16:
; CHECK-NEXT: st1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv8i16(<vscale x 8 x i16> %val, <vscale x 8 x i16> *%a, i32 2, <vscale x 8 x i1> %mask)
  ret void
}

define void @masked_store_nxv16i8(<vscale x 16 x i8> *%a, <vscale x 16 x i8> %val, <vscale x 16 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv16i8:
; CHECK-NEXT: st1b { z0.b }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv16i8(<vscale x 16 x i8> %val, <vscale x 16 x i8> *%a, i32 1, <vscale x 16 x i1> %mask)
  ret void
}

define void @masked_store_nxv2f64(<vscale x 2 x double> *%a, <vscale x 2 x double> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2f64:
; CHECK-NEXT: st1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2f64(<vscale x 2 x double> %val, <vscale x 2 x double> *%a, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv2f32(<vscale x 2 x float> *%a, <vscale x 2 x float> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2f32:
; CHECK-NEXT: st1w { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2f32(<vscale x 2 x float> %val, <vscale x 2 x float> *%a, i32 4, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv2f16(<vscale x 2 x half> *%a, <vscale x 2 x half> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2f16:
; CHECK-NEXT: st1h { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2f16(<vscale x 2 x half> %val, <vscale x 2 x half> *%a, i32 4, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv4f32(<vscale x 4 x float> *%a, <vscale x 4 x float> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4f32:
; CHECK-NEXT: st1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv4f32(<vscale x 4 x float> %val, <vscale x 4 x float> *%a, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_store_nxv4f16(<vscale x 4 x half> *%a, <vscale x 4 x half> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4f16:
; CHECK-NEXT: st1h { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv4f16(<vscale x 4 x half> %val, <vscale x 4 x half> *%a, i32 2, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_store_nxv8f16(<vscale x 8 x half> *%a, <vscale x 8 x half> %val, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8f16:
; CHECK-NEXT: st1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv8f16(<vscale x 8 x half> %val, <vscale x 8 x half> *%a, i32 2, <vscale x 8 x i1> %mask)
  ret void
}

declare <vscale x 2 x i64> @llvm.masked.load.nxv2i64(<vscale x 2 x i64>*, i32, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32(<vscale x 4 x i32>*, i32, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 8 x i16> @llvm.masked.load.nxv8i16(<vscale x 8 x i16>*, i32, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 16 x i8> @llvm.masked.load.nxv16i8(<vscale x 16 x i8>*, i32, <vscale x 16 x i1>, <vscale x 16 x i8>)

declare <vscale x 2 x double> @llvm.masked.load.nxv2f64(<vscale x 2 x double>*, i32, <vscale x 2 x i1>, <vscale x 2 x double>)
declare <vscale x 2 x float> @llvm.masked.load.nxv2f32(<vscale x 2 x float>*, i32, <vscale x 2 x i1>, <vscale x 2 x float>)
declare <vscale x 2 x half> @llvm.masked.load.nxv2f16(<vscale x 2 x half>*, i32, <vscale x 2 x i1>, <vscale x 2 x half>)
declare <vscale x 4 x float> @llvm.masked.load.nxv4f32(<vscale x 4 x float>*, i32, <vscale x 4 x i1>, <vscale x 4 x float>)
declare <vscale x 4 x half> @llvm.masked.load.nxv4f16(<vscale x 4 x half>*, i32, <vscale x 4 x i1>, <vscale x 4 x half>)
declare <vscale x 8 x half> @llvm.masked.load.nxv8f16(<vscale x 8 x half>*, i32, <vscale x 8 x i1>, <vscale x 8 x half>)

declare void @llvm.masked.store.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>*, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>*, i32, <vscale x 8 x i1>)
declare void @llvm.masked.store.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>*, i32, <vscale x 16 x i1>)

declare void @llvm.masked.store.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2f16(<vscale x 2 x half>, <vscale x 2 x half>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>*, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv4f16(<vscale x 4 x half>, <vscale x 4 x half>*, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>*, i32, <vscale x 8 x i1>)
