; RUN: opt -instcombine %s -S | FileCheck %s

; CHECK-LABEL: patatino
; CHECK-NEXT: ret <8 x i64*> undef
define <8 x i64*> @patatino() {
  %el = getelementptr i64, <8 x i64*> undef, <8 x i64> undef
  ret <8 x i64*> %el
}

; CHECK-LABEL: patatino2
; CHECK-NEXT: ret <8 x i64*> undef
define <8 x i64*> @patatino2() {
  %el = getelementptr inbounds i64, i64* undef, <8 x i64> undef
  ret <8 x i64*> %el
}

@block = global [64 x [8192 x i8]] zeroinitializer, align 1

; CHECK-LABEL:vectorindex
; CHECK-NEXT: ret <2 x i8*> getelementptr inbounds ([64 x [8192 x i8]], [64 x [8192 x i8]]* @block, <2 x i64> zeroinitializer, <2 x i64> <i64 0, i64 1>, <2 x i64> <i64 8192, i64 8192>)
define <2 x i8*> @vectorindex() {
  %1 = getelementptr inbounds [64 x [8192 x i8]], [64 x [8192 x i8]]* @block, i64 0, <2 x i64> <i64 0, i64 1>, i64 8192
  ret <2 x i8*> %1
}
