; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve -asm-verbose=0 < %s | FileCheck %s

;
; Masked Loads
;

define <vscale x 2 x i64> @masked_load_nxv2i64(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2i64:
; CHECK-NEXT: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64(ptr %a, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x i64> undef)
  ret <vscale x 2 x i64> %load
}

define <vscale x 4 x i32> @masked_load_nxv4i32(ptr %a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4i32:
; CHECK-NEXT: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32(ptr %a, i32 4, <vscale x 4 x i1> %mask, <vscale x 4 x i32> undef)
  ret <vscale x 4 x i32> %load
}

define <vscale x 8 x i16> @masked_load_nxv8i16(ptr %a, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8i16:
; CHECK-NEXT: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16(ptr %a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x i16> undef)
  ret <vscale x 8 x i16> %load
}

define <vscale x 16 x i8> @masked_load_nxv16i8(ptr %a, <vscale x 16 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv16i8:
; CHECK-NEXT: ld1b { z0.b }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8(ptr %a, i32 1, <vscale x 16 x i1> %mask, <vscale x 16 x i8> undef)
  ret <vscale x 16 x i8> %load
}

define <vscale x 2 x double> @masked_load_nxv2f64(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f64:
; CHECK-NEXT: ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x double> @llvm.masked.load.nxv2f64(ptr %a, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x double> undef)
  ret <vscale x 2 x double> %load
}

define <vscale x 2 x float> @masked_load_nxv2f32(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f32:
; CHECK-NEXT: ld1w { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x float> @llvm.masked.load.nxv2f32(ptr %a, i32 4, <vscale x 2 x i1> %mask, <vscale x 2 x float> undef)
  ret <vscale x 2 x float> %load
}

define <vscale x 2 x half> @masked_load_nxv2f16(ptr %a, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv2f16:
; CHECK-NEXT: ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x half> @llvm.masked.load.nxv2f16(ptr %a, i32 2, <vscale x 2 x i1> %mask, <vscale x 2 x half> undef)
  ret <vscale x 2 x half> %load
}

define <vscale x 2 x bfloat> @masked_load_nxv2bf16(ptr %a, <vscale x 2 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_load_nxv2bf16:
; CHECK-NEXT: ld1h { z0.d }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 2 x bfloat> @llvm.masked.load.nxv2bf16(ptr %a, i32 2, <vscale x 2 x i1> %mask, <vscale x 2 x bfloat> undef)
  ret <vscale x 2 x bfloat> %load
}

define <vscale x 4 x float> @masked_load_nxv4f32(ptr %a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4f32:
; CHECK-NEXT: ld1w { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x float> @llvm.masked.load.nxv4f32(ptr %a, i32 4, <vscale x 4 x i1> %mask, <vscale x 4 x float> undef)
  ret <vscale x 4 x float> %load
}

define <vscale x 4 x half> @masked_load_nxv4f16(ptr %a, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv4f16:
; CHECK-NEXT: ld1h { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x half> @llvm.masked.load.nxv4f16(ptr %a, i32 2, <vscale x 4 x i1> %mask, <vscale x 4 x half> undef)
  ret <vscale x 4 x half> %load
}

define <vscale x 4 x bfloat> @masked_load_nxv4bf16(ptr %a, <vscale x 4 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_load_nxv4bf16:
; CHECK-NEXT: ld1h { z0.s }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 4 x bfloat> @llvm.masked.load.nxv4bf16(ptr %a, i32 2, <vscale x 4 x i1> %mask, <vscale x 4 x bfloat> undef)
  ret <vscale x 4 x bfloat> %load
}

define <vscale x 8 x half> @masked_load_nxv8f16(ptr %a, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_load_nxv8f16:
; CHECK-NEXT: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x half> @llvm.masked.load.nxv8f16(ptr %a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x half> undef)
  ret <vscale x 8 x half> %load
}

define <vscale x 8 x bfloat> @masked_load_nxv8bf16(ptr %a, <vscale x 8 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked_load_nxv8bf16:
; CHECK-NEXT: ld1h { z0.h }, p0/z, [x0]
; CHECK-NEXT: ret
  %load = call <vscale x 8 x bfloat> @llvm.masked.load.nxv8bf16(ptr %a, i32 2, <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> undef)
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
  %wide.load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(ptr %in, i32 2, <vscale x 2 x i1> %mask, <vscale x 2 x i16> undef)
  ret <vscale x 2 x i16> %wide.load
}

;
; Masked Stores
;

define void @masked_store_nxv2i64(ptr %a, <vscale x 2 x i64> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2i64:
; CHECK-NEXT: st1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2i64(<vscale x 2 x i64> %val, ptr %a, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv4i32(ptr %a, <vscale x 4 x i32> %val, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv4i32:
; CHECK-NEXT: st1w { z0.s }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv4i32(<vscale x 4 x i32> %val, ptr %a, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_store_nxv8i16(ptr %a, <vscale x 8 x i16> %val, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv8i16:
; CHECK-NEXT: st1h { z0.h }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv8i16(<vscale x 8 x i16> %val, ptr %a, i32 2, <vscale x 8 x i1> %mask)
  ret void
}

define void @masked_store_nxv16i8(ptr %a, <vscale x 16 x i8> %val, <vscale x 16 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv16i8:
; CHECK-NEXT: st1b { z0.b }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv16i8(<vscale x 16 x i8> %val, ptr %a, i32 1, <vscale x 16 x i1> %mask)
  ret void
}

define void @masked_store_nxv2f64(ptr %a, <vscale x 2 x double> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2f64:
; CHECK-NEXT: st1d { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2f64(<vscale x 2 x double> %val, ptr %a, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_store_nxv2f32(ptr %a, <vscale x 2 x float> %val, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_store_nxv2f32:
; CHECK-NEXT: st1w { z0.d }, p0, [x0]
; CHECK-NEXT: ret
  call void @llvm.masked.store.nxv2f32(<vscale x 2 x float> %val, ptr %a, i32 4, <vscale x 2 x i1> %mask)
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

;
; Masked load store of pointer data type
;

; Pointer of integer type

define <vscale x 2 x ptr> @masked.load.nxv2p0i8(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0i8:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> undef)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0i16(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0i16:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> undef)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0i32(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0i32:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> undef)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0i64(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0i64:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> undef)
  ret <vscale x 2 x ptr> %v
}

; Pointer of floating-point type

define <vscale x 2 x ptr> @masked.load.nxv2p0bf16(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind #0 {
; CHECK-LABEL: masked.load.nxv2p0bf16:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> undef)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0f16(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0f16:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> undef)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0f32(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0f32:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> undef)
  ret <vscale x 2 x ptr> %v
}
define <vscale x 2 x ptr> @masked.load.nxv2p0f64(ptr %vector_ptr, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked.load.nxv2p0f64:
; CHECK-NEXT:    ld1d { z0.d }, p0/z, [x0]
; CHECK-NEXT:    ret
  %v = call <vscale x 2 x ptr> @llvm.masked.load.nxv2p0.p0(ptr %vector_ptr, i32 8, <vscale x 2 x i1> %mask, <vscale x 2 x ptr> undef)
  ret <vscale x 2 x ptr> %v
}

; Pointer of array type

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
