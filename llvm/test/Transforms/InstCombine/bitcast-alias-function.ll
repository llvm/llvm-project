; RUN: opt -S -instcombine -o - %s | FileCheck %s
target datalayout = "e-p:32:32:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v64:64:64-v128:128:128-a0:0:64"



; Cases that should be bitcast

; Test cast between scalars with same bit sizes
@alias_i32_to_f32 = alias float (float), bitcast (i32 (i32)* @func_i32 to float (float)*)

; Test cast between vectors with same number of elements and bit sizes
@alias_v2i32_to_v2f32 = alias <2 x float> (<2 x float>), bitcast (<2 x i32> (<2 x i32>)* @func_v2i32 to <2 x float> (<2 x float>)*)

; Test cast from vector to scalar with same number of bits
@alias_v2f32_to_i64 = alias <2 x float> (<2 x float>), bitcast (i64 (i64)* @func_i64 to <2 x float> (<2 x float>)*)

; Test cast from scalar to vector with same number of bits
@alias_i64_to_v2f32 = alias i64 (i64), bitcast (<2 x float> (<2 x float>)* @func_v2f32 to i64 (i64)*)

; Test cast between vectors of pointers
@alias_v2i32p_to_v2i64p = alias <2 x i64*> (<2 x i64*>), bitcast (<2 x i32*> (<2 x i32*>)* @func_v2i32p to <2 x i64*> (<2 x i64*>)*)


; Cases that should be invalid and unchanged

; Test cast between scalars with different bit sizes
@alias_i64_to_f32 = alias float (float), bitcast (i64 (i64)* @func_i64 to float (float)*)

; Test cast between vectors with different bit sizes but the
; same number of elements
@alias_v2i64_to_v2f32 = alias <2 x float> (<2 x float>), bitcast (<2 x i64> (<2 x i64>)* @func_v2i64 to <2 x float> (<2 x float>)*)

; Test cast between vectors with same number of bits and different
; numbers of elements
@alias_v2i32_to_v4f32 = alias <4 x float> (<4 x float>), bitcast (<2 x i32> (<2 x i32>)* @func_v2i32 to <4 x float> (<4 x float>)*)

; Test cast between scalar and vector with different number of bits
@alias_i64_to_v4f32 = alias i64 (i64), bitcast (<4 x float> (<4 x float>)* @func_v4f32 to i64 (i64)*)

; Test cast between vector and scalar with different number of bits
@alias_v4f32_to_i64 = alias <4 x float> (<4 x float>), bitcast (i64 (i64)* @func_i64 to <4 x float> (<4 x float>)*)

; Test cast from scalar to vector of pointers with same number of bits
; We don't know the pointer size at this point, so this can't be done
@alias_i64_to_v2i32p = alias i64 (i64), bitcast (<2 x i32*> (<2 x i32*>)* @func_v2i32p to i64 (i64)*)

; Test cast between vector of pointers and scalar with different number of bits
@alias_v4i32p_to_i64 = alias <4 x i32*> (<4 x i32*>), bitcast (i64 (i64)* @func_i64 to <4 x i32*> (<4 x i32*>)*)



define internal <2 x i32> @func_v2i32(<2 x i32> %v) noinline nounwind {
entry:
  ret <2 x i32> %v
}

define internal <2 x float> @func_v2f32(<2 x float> %v) noinline nounwind {
entry:
  ret <2 x float> %v
}

define internal <4 x float> @func_v4f32(<4 x float> %v) noinline nounwind {
entry:
  ret <4 x float> %v
}

define internal i32 @func_i32(i32 %v) noinline nounwind {
entry:
  ret i32 %v
}

define internal i64 @func_i64(i64 %v) noinline nounwind {
entry:
  ret i64 %v
}

define internal <2 x i64> @func_v2i64(<2 x i64> %v) noinline nounwind {
entry:
  ret <2 x i64> %v
}

define internal <2 x i32*> @func_v2i32p(<2 x i32*> %v) noinline nounwind {
entry:
  ret <2 x i32*> %v
}

; Valid cases, only bitcast for argument / return type and call underlying function

; Sizes match, should only bitcast
define void @bitcast_alias_scalar(float* noalias %source, float* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_scalar
; CHECK: bitcast float* %source to i32*
; CHECK: load i32, i32*
; CHECK-NOT: fptoui
; CHECK-NOT: uitofp
; CHECK: bitcast float* %dest to i32*
; CHECK: store i32
  %tmp = load float, float* %source, align 8
  %call = call float @alias_i32_to_f32(float %tmp) nounwind
  store float %call, float* %dest, align 8
  ret void
}

; Sizes match, should only bitcast
define void @bitcast_alias_vector(<2 x float>* noalias %source, <2 x float>* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_vector
; CHECK: bitcast <2 x float>* %source to <2 x i32>*
; CHECK: load <2 x i32>, <2 x i32>*
; CHECK-NOT: fptoui
; CHECK-NOT: uitofp
; CHECK: bitcast <2 x float>* %dest to <2 x i32>*
; CHECK: store <2 x i32>
  %tmp = load <2 x float>, <2 x float>* %source, align 8
  %call = call <2 x float> @alias_v2i32_to_v2f32(<2 x float> %tmp) nounwind
  store <2 x float> %call, <2 x float>* %dest, align 8
  ret void
}

; Sizes match, should only bitcast
define void @bitcast_alias_vector_scalar_same_size(<2 x float>* noalias %source, <2 x float>* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_vector_scalar_same_size
; CHECK: bitcast <2 x float>* %source to i64*
; CHECK: load i64, i64*
; CHECK: %call = call i64 @func_i64
; CHECK: bitcast <2 x float>* %dest to i64*
; CHECK: store i64
  %tmp = load <2 x float>, <2 x float>* %source, align 8
  %call = call <2 x float> @alias_v2f32_to_i64(<2 x float> %tmp) nounwind
  store <2 x float> %call, <2 x float>* %dest, align 8
  ret void
}

define void @bitcast_alias_scalar_vector_same_size(i64* noalias %source, i64* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_scalar_vector_same_size
; CHECK: bitcast i64* %source to <2 x float>*
; CHECK: load <2 x float>, <2 x float>*
; CHECK: call <2 x float> @func_v2f32
; CHECK: bitcast i64* %dest to <2 x float>*
; CHECK: store <2 x float>
  %tmp = load i64, i64* %source, align 8
  %call = call i64 @alias_i64_to_v2f32(i64 %tmp) nounwind
  store i64 %call, i64* %dest, align 8
  ret void
}

define void @bitcast_alias_vector_ptrs_same_size(<2 x i64*>* noalias %source, <2 x i64*>* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_vector_ptrs_same_size
; CHECK: bitcast <2 x i64*>* %source to <2 x i32*>*
; CHECK: load <2 x i32*>, <2 x i32*>*
; CHECK: call <2 x i32*> @func_v2i32p
; CHECK: bitcast <2 x i64*>* %dest to <2 x i32*>*
; CHECK: store <2 x i32*>
  %tmp = load <2 x i64*>, <2 x i64*>* %source, align 8
  %call = call <2 x i64*> @alias_v2i32p_to_v2i64p(<2 x i64*> %tmp) nounwind
  store <2 x i64*> %call, <2 x i64*>* %dest, align 8
  ret void
}

; Invalid cases:

define void @bitcast_alias_mismatch_scalar_size(float* noalias %source, float* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_mismatch_scalar_size
; CHECK-NOT: fptoui
; CHECK: @alias_i64_to_f32
; CHECK-NOT: uitofp
  %tmp = load float, float* %source, align 8
  %call = call float @alias_i64_to_f32(float %tmp) nounwind
  store float %call, float* %dest, align 8
  ret void
}

define void @bitcast_alias_mismatch_vector_element_and_bit_size(<2 x float>* noalias %source, <2 x float>* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_mismatch_vector_element_and_bit_size
; CHECK-NOT: fptoui <2 x float> %tmp to <2 x i64>
; CHECK: @alias_v2i64_to_v2f32
; CHECK-NOT: uitofp <2 x i64> %call to <2 x float>
  %tmp = load <2 x float>, <2 x float>* %source, align 8
  %call = call <2 x float> @alias_v2i64_to_v2f32(<2 x float> %tmp) nounwind
  store <2 x float> %call, <2 x float>* %dest, align 8
  ret void
}

define void @bitcast_alias_vector_mismatched_number_elements(<4 x float>* noalias %source, <4 x float>* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_vector_mismatched_number_elements
; CHECK:  %call = call <4 x float> @alias_v2i32_to_v4f32
  %tmp = load <4 x float>, <4 x float>* %source, align 8
  %call = call <4 x float> @alias_v2i32_to_v4f32(<4 x float> %tmp) nounwind
  store <4 x float> %call, <4 x float>* %dest, align 8
  ret void
}

define void @bitcast_alias_vector_scalar_mismatched_bit_size(<4 x float>* noalias %source, <4 x float>* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_vector_scalar_mismatched_bit_size
; CHECK:  %call = call <4 x float> @alias_v4f32_to_i64
  %tmp = load <4 x float>, <4 x float>* %source, align 8
  %call = call <4 x float> @alias_v4f32_to_i64(<4 x float> %tmp) nounwind
  store <4 x float> %call, <4 x float>* %dest, align 8
  ret void
}

define void @bitcast_alias_vector_ptrs_scalar_mismatched_bit_size(<4 x i32*>* noalias %source, <4 x i32*>* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_vector_ptrs_scalar_mismatched_bit_size
; CHECK: @alias_v4i32p_to_i64
  %tmp = load <4 x i32*>, <4 x i32*>* %source, align 8
  %call = call <4 x i32*> @alias_v4i32p_to_i64(<4 x i32*> %tmp) nounwind
  store <4 x i32*> %call, <4 x i32*>* %dest, align 8
  ret void
}

define void @bitcast_alias_scalar_vector_ptrs_same_size(i64* noalias %source, i64* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_scalar_vector_ptrs_same_size
; CHECK: @alias_i64_to_v2i32p
  %tmp = load i64, i64* %source, align 8
  %call = call i64 @alias_i64_to_v2i32p(i64 %tmp) nounwind
  store i64 %call, i64* %dest, align 8
  ret void
}

define void @bitcast_alias_scalar_vector_mismatched_bit_size(i64* noalias %source, i64* noalias %dest) nounwind {
entry:
; CHECK-LABEL: @bitcast_alias_scalar_vector_mismatched_bit_size
; CHECK: call i64 @alias_i64_to_v4f32
  %tmp = load i64, i64* %source, align 8
  %call = call i64 @alias_i64_to_v4f32(i64 %tmp) nounwind
  store i64 %call, i64* %dest, align 8
  ret void
}

