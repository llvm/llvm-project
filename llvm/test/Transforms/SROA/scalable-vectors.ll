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
