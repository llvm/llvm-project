; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

define <4 x i64> @vector_ptrtoint({<2 x i32*>, <2 x i32*>} %x) {
; CHECK-LABEL: @vector_ptrtoint
  %a = alloca {<2 x i32*>, <2 x i32*>}
; CHECK-NOT: alloca

  store {<2 x i32*>, <2 x i32*>} %x, {<2 x i32*>, <2 x i32*>}* %a
; CHECK-NOT: store

  %cast = bitcast {<2 x i32*>, <2 x i32*>}* %a to <4 x i64>*
  %vec = load <4 x i64>, <4 x i64>* %cast
; CHECK-NOT: load
; CHECK: ptrtoint

  ret <4 x i64> %vec
}

define <4 x i32*> @vector_inttoptr({<2 x i64>, <2 x i64>} %x) {
; CHECK-LABEL: @vector_inttoptr
  %a = alloca {<2 x i64>, <2 x i64>}
; CHECK-NOT: alloca

  store {<2 x i64>, <2 x i64>} %x, {<2 x i64>, <2 x i64>}* %a
; CHECK-NOT: store

  %cast = bitcast {<2 x i64>, <2 x i64>}* %a to <4 x i32*>*
  %vec = load <4 x i32*>, <4 x i32*>* %cast
; CHECK-NOT: load
; CHECK: inttoptr

  ret <4 x i32*> %vec
}

define <2 x i64> @vector_ptrtointbitcast({<1 x i32*>, <1 x i32*>} %x) {
; CHECK-LABEL: @vector_ptrtointbitcast
  %a = alloca {<1 x i32*>, <1 x i32*>}
; CHECK-NOT: alloca

  store {<1 x i32*>, <1 x i32*>} %x, {<1 x i32*>, <1 x i32*>}* %a
; CHECK-NOT: store

  %cast = bitcast {<1 x i32*>, <1 x i32*>}* %a to <2 x i64>*
  %vec = load <2 x i64>, <2 x i64>* %cast
; CHECK-NOT: load
; CHECK: ptrtoint
; CHECK: bitcast
; CHECK: ptrtoint
; CHECK: bitcast

  ret <2 x i64> %vec
}
