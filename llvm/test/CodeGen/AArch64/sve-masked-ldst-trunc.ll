; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve -asm-verbose=0 < %s | FileCheck %s

;
; Masked Stores
;

define void @masked_trunc_store_nxv2i8(ptr %a, <vscale x 2 x i64> %val, ptr %b, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv2i8:
; CHECK-NEXT: st1b { z0.d }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i8>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %trunc, ptr %b, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv2i16(ptr %a, <vscale x 2 x i64> %val, ptr %b, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv2i16:
; CHECK-NEXT: st1h { z0.d }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i16>
  call void @llvm.masked.store.nxv2i16(<vscale x 2 x i16> %trunc, ptr %b, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv2i32(ptr %a, <vscale x 2 x i64> %val, ptr %b, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv2i32:
; CHECK-NEXT: st1w { z0.d }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i32>
  call void @llvm.masked.store.nxv2i32(<vscale x 2 x i32> %trunc, ptr %b, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv4i8(ptr %a, <vscale x 4 x i32> %val, ptr %b, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv4i8:
; CHECK-NEXT: st1b { z0.s }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 4 x i32> %val to <vscale x 4 x i8>
  call void @llvm.masked.store.nxv4i8(<vscale x 4 x i8> %trunc, ptr %b, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv4i16(ptr %a, <vscale x 4 x i32> %val, ptr %b, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv4i16:
; CHECK-NEXT: st1h { z0.s }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 4 x i32> %val to <vscale x 4 x i16>
  call void @llvm.masked.store.nxv4i16(<vscale x 4 x i16> %trunc, ptr %b, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv8i8(ptr %a, <vscale x 8 x i16> %val, ptr %b, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv8i8:
; CHECK-NEXT: st1b { z0.h }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 8 x i16> %val to <vscale x 8 x i8>
  call void @llvm.masked.store.nxv8i8(<vscale x 8 x i8> %trunc, ptr %b, i32 2, <vscale x 8 x i1> %mask)
  ret void
}

declare void @llvm.masked.store.nxv2i8(<vscale x 2 x i8>, ptr, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2i16(<vscale x 2 x i16>, ptr, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2i32(<vscale x 2 x i32>, ptr, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv4i8(<vscale x 4 x i8>, ptr, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv4i16(<vscale x 4 x i16>, ptr, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv8i8(<vscale x 8 x i8>, ptr, i32, <vscale x 8 x i1>)
