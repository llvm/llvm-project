; RUN: opt -p lower-matrix-intrinsics -matrix-print-after-transpose-opt -disable-output -S %s 2>&1 | FileCheck %s

; REQUIRES: asserts

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; FIXME: Lifted transpose dimensions are incorrect.
define <6 x double> @lift_through_add_matching_transpose_dimensions(<6 x double> %a, <6 x double> %b) {
; CHECK-LABEL:  define <6 x double> @lift_through_add_matching_transpose_dimensions(<6 x double> %a, <6 x double> %b) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.+]] = fadd <6 x double> %a, %b
; CHECK-NEXT:    [[T:%.+]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[A]], i32 3, i32 2)
; CHECK-NEXT:    ret <6 x double> [[T]]
;
entry:
  %a.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %a, i32 3, i32 2)
  %b.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %b, i32 3, i32 2)
  %add = fadd <6 x double> %a.t, %b.t
  ret <6 x double> %add
}

define <6 x double> @lift_through_add_matching_transpose_dimensions_ops_also_have_shape_info(ptr %a.ptr, ptr %b.ptr) {
; CHECK-LABEL: define <6 x double> @lift_through_add_matching_transpose_dimensions_ops_also_have_shape_info(ptr %a.ptr, ptr %b.ptr)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.+]] = load <6 x double>, ptr %a.ptr
; CHECK-NEXT:    [[B:%.+]] = load <6 x double>, ptr %b.ptr
; CHECK-NEXT:    [[ADD:%.+]] = fadd <6 x double> [[A]], [[B]]
; CHECK-NEXT:    [[T:%.+]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[ADD]], i32 3, i32 2)
; CHECK-NEXT:    ret <6 x double> [[T]]
;
entry:
  %a = load <6 x double>, ptr %a.ptr
  %b = load <6 x double>, ptr %b.ptr
  %a.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %a, i32 3, i32 2)
  %b.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %b, i32 3, i32 2)
  %add = fadd <6 x double> %a.t, %b.t
  ret <6 x double> %add
}

define <6 x double> @lift_through_add_mismatching_dimensions_1(<6 x double> %a, <6 x double> %b) {
; CHECK-LABEL:  define <6 x double> @lift_through_add_mismatching_dimensions_1(<6 x double> %a, <6 x double> %b) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.+]] = fadd <6 x double> %a, %b
; CHECK-NEXT:    [[T:%.+]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[A]], i32 1, i32 6)
; CHECK-NEXT:    ret <6 x double> [[T]]
;
entry:
  %a.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %a, i32 1, i32 6)
  %b.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %b, i32 3, i32 2)
  %add = fadd <6 x double> %a.t, %b.t
  ret <6 x double> %add
}

define <6 x double> @lift_through_add_mismatching_dimensions_1_transpose_dimensions_ops_also_have_shape_info(ptr %a.ptr, ptr %b.ptr) {
; CHECK-LABEL: define <6 x double> @lift_through_add_mismatching_dimensions_1_transpose_dimensions_ops_also_have_shape_info(ptr %a.ptr, ptr %b.ptr)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.+]] = load <6 x double>, ptr %a.ptr
; CHECK-NEXT:    [[B:%.+]] = load <6 x double>, ptr %b.ptr
; CHECK-NEXT:    [[ADD:%.+]] = fadd <6 x double> [[A]], [[B]]
; CHECK-NEXT:    [[T:%.+]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[ADD]], i32 1, i32 6)
; CHECK-NEXT:    ret <6 x double> [[T]]
;
entry:
  %a = load <6 x double>, ptr %a.ptr
  %b = load <6 x double>, ptr %b.ptr
  %a.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %a, i32 1, i32 6)
  %b.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %b, i32 3, i32 2)
  %add = fadd <6 x double> %a.t, %b.t
  ret <6 x double> %add
}

define <6 x double> @lift_through_add_mismatching_dimensions_2(<6 x double> %a, <6 x double> %b) {
; CHECK-LABEL:  define <6 x double> @lift_through_add_mismatching_dimensions_2(<6 x double> %a, <6 x double> %b) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.+]] = fadd <6 x double> %a, %b
; CHECK-NEXT:    [[T:%.+]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[A]], i32 3, i32 2)
; CHECK-NEXT:    ret <6 x double> [[T]]
;

entry:
  %a.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %a, i32 3, i32 2)
  %b.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %b, i32 6, i32 1)
  %add = fadd <6 x double> %a.t, %b.t
  ret <6 x double> %add
}

define <6 x double> @lift_through_add_mismatching_dimensions_2_transpose_dimensions_ops_also_have_shape_info(ptr %a.ptr, ptr %b.ptr) {
; CHECK-LABEL: define <6 x double> @lift_through_add_mismatching_dimensions_2_transpose_dimensions_ops_also_have_shape_info(ptr %a.ptr, ptr %b.ptr)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.+]] = load <6 x double>, ptr %a.ptr
; CHECK-NEXT:    [[B:%.+]] = load <6 x double>, ptr %b.ptr
; CHECK-NEXT:    [[ADD:%.+]] = fadd <6 x double> [[A]], [[B]]
; CHECK-NEXT:    [[T:%.+]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[ADD]], i32 3, i32 2)
; CHECK-NEXT:    ret <6 x double> [[T]]
;
entry:
  %a = load <6 x double>, ptr %a.ptr
  %b = load <6 x double>, ptr %b.ptr
  %a.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %a, i32 3, i32 2)
  %b.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %b, i32 6, i32 1)
  %add = fadd <6 x double> %a.t, %b.t
  ret <6 x double> %add
}

define <9 x double> @lift_through_multiply(<6 x double> %a, <6 x double> %b) {
; CHECK-LABEL: define <9 x double> @lift_through_multiply(<6 x double> %a, <6 x double> %b) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MUL:%.+]] = call <9 x double> @llvm.matrix.multiply.v9f64.v6f64.v6f64(<6 x double> %b, <6 x double> %a, i32 3, i32 2, i32 3)
; CHECK-NEXT:    [[T:%.+]] = call <9 x double> @llvm.matrix.transpose.v9f64(<9 x double> [[MUL]], i32 3, i32 3)
; CHECK-NEXT:   ret <9 x double> [[T]]
;
entry:
  %a.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %a, i32 3, i32 2)
  %b.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %b, i32 2, i32 3)
  %mul = call <9 x double> @llvm.matrix.multiply.v9f64.v6f64(<6 x double> %a.t, <6 x double> %b.t, i32 3, i32 2 , i32 3)
  ret <9 x double> %mul
}

define <6 x double> @lift_through_multiply_2(<6 x double> %a, <4 x double> %b) {
; CHECK-LABEL: define <6 x double> @lift_through_multiply_2(<6 x double> %a, <4 x double> %b) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MUL:%.+]] = call <6 x double> @llvm.matrix.multiply.v6f64.v4f64.v6f64(<4 x double> %b, <6 x double> %a, i32 2, i32 2, i32 3)
; CHECK-NEXT:    [[T:%.+]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[MUL]], i32 2, i32 3)
; CHECK-NEXT:    ret <6 x double> [[T]]
;
entry:
  %a.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %a, i32 3, i32 2)
  %b.t = call <4 x double> @llvm.matrix.transpose.v4f64(<4 x double> %b, i32 2, i32 2)
  %mul = call <6 x double> @llvm.matrix.multiply.v6f64.v6f64.v4f64(<6 x double> %a.t, <4 x double> %b.t, i32 3, i32 2 , i32 2)
  ret <6 x double> %mul
}

define <6 x double> @lift_through_multiply_3(<4 x double> %a, <6 x double> %b) {
; CHECK-LABEL: define <6 x double> @lift_through_multiply_3(<4 x double> %a, <6 x double> %b) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MUL:%.+]] = call <6 x double> @llvm.matrix.multiply.v6f64.v6f64.v4f64(<6 x double> %b, <4 x double> %a, i32 3, i32 2, i32 2)
; CHECK-NEXT:    [[T:%.+]] = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> [[MUL]], i32 3, i32 2)
; CHECK-NEXT:    ret <6 x double> [[T]]
;
entry:
  %a.t = call <4 x double> @llvm.matrix.transpose.v4f64(<4 x double> %a, i32 2, i32 2)
  %b.t = call <6 x double> @llvm.matrix.transpose.v6f64(<6 x double> %b, i32 2, i32 3)
  %mul = call <6 x double> @llvm.matrix.multiply.v6f64.v4f64.v6f64(<4 x double> %a.t, <6 x double> %b.t, i32 2, i32 2 , i32 3)
  ret <6 x double> %mul
}

declare <6 x double> @llvm.matrix.transpose.v6f64.v6f64(<6 x double>, i32, i32)
declare <4 x double> @llvm.matrix.transpose.v4f64.v4f64(<4 x double>, i32, i32)
declare <9 x double> @llvm.matrix.multiply.v9f64.v6f64(<6 x double>, <6 x double>, i32, i32, i32)
declare <6 x double> @llvm.matrix.multiply.v6f64.v6f64.v4f64(<6 x double>, <4 x double>, i32, i32, i32)
declare <6 x double> @llvm.matrix.multiply.v6f64.v6f64.v6f64(<6 x double>, <4 x double>, i32, i32, i32)
