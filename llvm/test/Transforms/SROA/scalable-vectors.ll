; RUN: opt < %s -sroa -S | FileCheck %s
; RUN: opt < %s -passes=sroa -S | FileCheck %s

; This test checks that SROA runs mem2reg on scalable vectors.

define <vscale x 16 x i1> @alloca_nxv16i1(<vscale x 16 x i1> %pg) {
; CHECK-LABEL: alloca_nxv16i1
; CHECK-NEXT: ret <vscale x 16 x i1> %pg
  %pg.addr = alloca <vscale x 16 x i1>
  store <vscale x 16 x i1> %pg, <vscale x 16 x i1>* %pg.addr
  %1 = load <vscale x 16 x i1>, <vscale x 16 x i1>* %pg.addr
  ret <vscale x 16 x i1> %1
}

define <vscale x 16 x i8> @alloca_nxv16i8(<vscale x 16 x i8> %vec) {
; CHECK-LABEL: alloca_nxv16i8
; CHECK-NEXT: ret <vscale x 16 x i8> %vec
  %vec.addr = alloca <vscale x 16 x i8>
  store <vscale x 16 x i8> %vec, <vscale x 16 x i8>* %vec.addr
  %1 = load <vscale x 16 x i8>, <vscale x 16 x i8>* %vec.addr
  ret <vscale x 16 x i8> %1
}

; Test scalable alloca that can't be promoted. Mem2Reg only considers
; non-volatile loads and stores for promotion.
define <vscale x 16 x i8> @unpromotable_alloca(<vscale x 16 x i8> %vec) {
; CHECK-LABEL: unpromotable_alloca
; CHECK-NEXT: %vec.addr = alloca <vscale x 16 x i8>
; CHECK-NEXT: store volatile <vscale x 16 x i8> %vec, <vscale x 16 x i8>* %vec.addr
; CHECK-NEXT: %1 = load volatile <vscale x 16 x i8>, <vscale x 16 x i8>* %vec.addr
; CHECK-NEXT: ret <vscale x 16 x i8> %1
  %vec.addr = alloca <vscale x 16 x i8>
  store volatile <vscale x 16 x i8> %vec, <vscale x 16 x i8>* %vec.addr
  %1 = load volatile <vscale x 16 x i8>, <vscale x 16 x i8>* %vec.addr
  ret <vscale x 16 x i8> %1
}

; Test we bail out when using an alloca of a fixed-length vector (VLS) that was
; bitcasted to a scalable vector.
define <vscale x 4 x i32> @cast_alloca_to_svint32_t(<vscale x 4 x i32> %type.coerce) {
; CHECK-LABEL: cast_alloca_to_svint32_t
; CHECK-NEXT: %type = alloca <16 x i32>, align 64
; CHECK-NEXT: %type.addr = alloca <16 x i32>, align 64
; CHECK-NEXT: %1 = bitcast <16 x i32>* %type to <vscale x 4 x i32>*
; CHECK-NEXT: store <vscale x 4 x i32> %type.coerce, <vscale x 4 x i32>* %1, align 16
; CHECK-NEXT: %type1 = load <16 x i32>, <16 x i32>* %type, align 64
; CHECK-NEXT: store <16 x i32> %type1, <16 x i32>* %type.addr, align 64
; CHECK-NEXT: %2 = load <16 x i32>, <16 x i32>* %type.addr, align 64
; CHECK-NEXT: %3 = bitcast <16 x i32>* %type.addr to <vscale x 4 x i32>*
; CHECK-NEXT: %4 = load <vscale x 4 x i32>, <vscale x 4 x i32>* %3, align 16
; CHECK-NEXT: ret <vscale x 4 x i32> %4
  %type = alloca <16 x i32>
  %type.addr = alloca <16 x i32>
  %1 = bitcast <16 x i32>* %type to <vscale x 4 x i32>*
  store <vscale x 4 x i32> %type.coerce, <vscale x 4 x i32>* %1
  %type1 = load <16 x i32>, <16 x i32>* %type
  store <16 x i32> %type1, <16 x i32>* %type.addr
  %2 = load <16 x i32>, <16 x i32>* %type.addr
  %3 = bitcast <16 x i32>* %type.addr to <vscale x 4 x i32>*
  %4 = load <vscale x 4 x i32>, <vscale x 4 x i32>* %3
  ret <vscale x 4 x i32> %4
}

; When casting from VLA to VLS via memory check we bail out when producing a
; GEP where the element type is a scalable vector.
define <vscale x 4 x i32> @cast_alloca_from_svint32_t() {
; CHECK-LABEL: cast_alloca_from_svint32_t
; CHECK-NEXT: %retval.coerce = alloca <vscale x 4 x i32>, align 16
; CHECK-NEXT: %1 = bitcast <vscale x 4 x i32>* %retval.coerce to i8*
; CHECK-NEXT: %retval.0..sroa_cast = bitcast i8* %1 to <16 x i32>*
; CHECK-NEXT: store <16 x i32> undef, <16 x i32>* %retval.0..sroa_cast, align 16
; CHECK-NEXT: %2 = load <vscale x 4 x i32>, <vscale x 4 x i32>* %retval.coerce, align 16
; CHECK-NEXT: ret <vscale x 4 x i32> %2
  %retval = alloca <16 x i32>
  %retval.coerce = alloca <vscale x 4 x i32>
  %1 = bitcast <vscale x 4 x i32>* %retval.coerce to i8*
  %2 = bitcast <16 x i32>* %retval to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %1, i8* align 16 %2, i64 64, i1 false)
  %3 = load <vscale x 4 x i32>, <vscale x 4 x i32>* %retval.coerce
  ret <vscale x 4 x i32> %3
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
