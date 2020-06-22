; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve -asm-verbose=0 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; Masked Stores
;

define void @masked_trunc_store_nxv2i8(<vscale x 2 x i64> *%a, <vscale x 2 x i64> %val, <vscale x 2 x i8> *%b, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv2i8:
; CHECK-NEXT: st1b { z0.d }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i8>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %trunc, <vscale x 2 x i8> *%b, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv2i16(<vscale x 2 x i64> *%a, <vscale x 2 x i64> %val, <vscale x 2 x i16> *%b, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv2i16:
; CHECK-NEXT: st1h { z0.d }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i16>
  call void @llvm.masked.store.nxv2i16(<vscale x 2 x i16> %trunc, <vscale x 2 x i16> *%b, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv2i32(<vscale x 2 x i64> *%a, <vscale x 2 x i64> %val, <vscale x 2 x i32> *%b, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv2i32:
; CHECK-NEXT: st1w { z0.d }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i32>
  call void @llvm.masked.store.nxv2i32(<vscale x 2 x i32> %trunc, <vscale x 2 x i32> *%b, i32 8, <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv4i8(<vscale x 4 x i32> *%a, <vscale x 4 x i32> %val, <vscale x 4 x i8> *%b, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv4i8:
; CHECK-NEXT: st1b { z0.s }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 4 x i32> %val to <vscale x 4 x i8>
  call void @llvm.masked.store.nxv4i8(<vscale x 4 x i8> %trunc, <vscale x 4 x i8> *%b, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv4i16(<vscale x 4 x i32> *%a, <vscale x 4 x i32> %val, <vscale x 4 x i16> *%b, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv4i16:
; CHECK-NEXT: st1h { z0.s }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 4 x i32> %val to <vscale x 4 x i16>
  call void @llvm.masked.store.nxv4i16(<vscale x 4 x i16> %trunc, <vscale x 4 x i16> *%b, i32 4, <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_trunc_store_nxv8i8(<vscale x 8 x i16> *%a, <vscale x 8 x i16> %val, <vscale x 8 x i8> *%b, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: masked_trunc_store_nxv8i8:
; CHECK-NEXT: st1b { z0.h }, p0, [x1]
; CHECK-NEXT: ret
  %trunc = trunc <vscale x 8 x i16> %val to <vscale x 8 x i8>
  call void @llvm.masked.store.nxv8i8(<vscale x 8 x i8> %trunc, <vscale x 8 x i8> *%b, i32 2, <vscale x 8 x i1> %mask)
  ret void
}

declare void @llvm.masked.store.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i8>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i16>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv4i8(<vscale x 4 x i8>, <vscale x 4 x i8>*, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i16>*, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv8i8(<vscale x 8 x i8>, <vscale x 8 x i8>*, i32, <vscale x 8 x i1>)
