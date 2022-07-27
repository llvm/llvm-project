; RUN: opt < %s -passes=sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

define <4 x i64> @vector_ptrtoint({<2 x ptr>, <2 x ptr>} %x) {
; CHECK-LABEL: @vector_ptrtoint
  %a = alloca {<2 x ptr>, <2 x ptr>}
; CHECK-NOT: alloca

  store {<2 x ptr>, <2 x ptr>} %x, ptr %a
; CHECK-NOT: store

  %vec = load <4 x i64>, ptr %a
; CHECK-NOT: load
; CHECK: ptrtoint

  ret <4 x i64> %vec
}

define <4 x ptr> @vector_inttoptr({<2 x i64>, <2 x i64>} %x) {
; CHECK-LABEL: @vector_inttoptr
  %a = alloca {<2 x i64>, <2 x i64>}
; CHECK-NOT: alloca

  store {<2 x i64>, <2 x i64>} %x, ptr %a
; CHECK-NOT: store

  %vec = load <4 x ptr>, ptr %a
; CHECK-NOT: load
; CHECK: inttoptr

  ret <4 x ptr> %vec
}

define <2 x i64> @vector_ptrtointbitcast({<1 x ptr>, <1 x ptr>} %x) {
; CHECK-LABEL: @vector_ptrtointbitcast(
  %a = alloca {<1 x ptr>, <1 x ptr>}
; CHECK-NOT: alloca

  store {<1 x ptr>, <1 x ptr>} %x, ptr %a
; CHECK-NOT: store

  %vec = load <2 x i64>, ptr %a
; CHECK-NOT: load
; CHECK: ptrtoint
; CHECK: bitcast
; CHECK: ptrtoint
; CHECK: bitcast

  ret <2 x i64> %vec
}

define <2 x ptr> @vector_inttoptrbitcast_vector({<16 x i8>, <16 x i8>} %x) {
; CHECK-LABEL: @vector_inttoptrbitcast_vector(
  %a = alloca {<16 x i8>, <16 x i8>}
; CHECK-NOT: alloca

  store {<16 x i8>, <16 x i8>} %x, ptr %a
; CHECK-NOT: store

  %vec = load <2 x ptr>, ptr %a
; CHECK-NOT: load
; CHECK: extractvalue
; CHECK: extractvalue
; CHECK: bitcast
; CHECK: inttoptr

  ret <2 x ptr> %vec
}

define <16 x i8> @vector_ptrtointbitcast_vector({<2 x ptr>, <2 x ptr>} %x) {
; CHECK-LABEL: @vector_ptrtointbitcast_vector(
  %a = alloca {<2 x ptr>, <2 x ptr>}
; CHECK-NOT: alloca

  store {<2 x ptr>, <2 x ptr>} %x, ptr %a
; CHECK-NOT: store

  %vec = load <16 x i8>, ptr %a
; CHECK-NOT: load
; CHECK: extractvalue
; CHECK: ptrtoint
; CHECK: bitcast
; CHECK: extractvalue

  ret <16 x i8> %vec
}
