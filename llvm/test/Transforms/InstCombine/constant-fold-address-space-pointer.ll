; RUN: opt -S -instcombine %s -o - | FileCheck %s
target datalayout = "e-p:32:32:32-p1:64:64:64-p2:8:8:8-p3:16:16:16-p4:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"

@g = addrspace(3) global i32 89

@const_zero_i8_as1 = addrspace(1) constant i8 0
@const_zero_i32_as1 = addrspace(1) constant i32 0

@const_zero_i8_as2 = addrspace(2) constant i8 0
@const_zero_i32_as2 = addrspace(2) constant i32 0

@const_zero_i8_as3 = addrspace(3) constant i8 0
@const_zero_i32_as3 = addrspace(3) constant i32 0

; Test constant folding of inttoptr (ptrtoint constantexpr)
; The intermediate integer size is the same as the pointer size
define i32 addrspace(3)* @test_constant_fold_inttoptr_as_pointer_same_size() {
; CHECK-LABEL: @test_constant_fold_inttoptr_as_pointer_same_size(
; CHECK-NEXT: ret i32 addrspace(3)* @const_zero_i32_as3
  %x = ptrtoint i32 addrspace(3)* @const_zero_i32_as3 to i32
  %y = inttoptr i32 %x to i32 addrspace(3)*
  ret i32 addrspace(3)* %y
}

; The intermediate integer size is larger than the pointer size
define i32 addrspace(2)* @test_constant_fold_inttoptr_as_pointer_smaller() {
; CHECK-LABEL: @test_constant_fold_inttoptr_as_pointer_smaller(
; CHECK-NEXT: ret i32 addrspace(2)* @const_zero_i32_as2
  %x = ptrtoint i32 addrspace(2)* @const_zero_i32_as2 to i16
  %y = inttoptr i16 %x to i32 addrspace(2)*
  ret i32 addrspace(2)* %y
}

; Different address spaces that are the same size, but they are
; different so nothing should happen
define i32 addrspace(4)* @test_constant_fold_inttoptr_as_pointer_smaller_different_as() {
; CHECK-LABEL: @test_constant_fold_inttoptr_as_pointer_smaller_different_as(
; CHECK-NEXT: ret i32 addrspace(4)* inttoptr (i16 ptrtoint (i32 addrspace(3)* @const_zero_i32_as3 to i16) to i32 addrspace(4)*)
  %x = ptrtoint i32 addrspace(3)* @const_zero_i32_as3 to i16
  %y = inttoptr i16 %x to i32 addrspace(4)*
  ret i32 addrspace(4)* %y
}

; Make sure we don't introduce a bitcast between different sized
; address spaces when folding this
define i32 addrspace(2)* @test_constant_fold_inttoptr_as_pointer_smaller_different_size_as() {
; CHECK-LABEL: @test_constant_fold_inttoptr_as_pointer_smaller_different_size_as(
; CHECK-NEXT: ret i32 addrspace(2)* inttoptr (i32 ptrtoint (i32 addrspace(3)* @const_zero_i32_as3 to i32) to i32 addrspace(2)*)
  %x = ptrtoint i32 addrspace(3)* @const_zero_i32_as3 to i32
  %y = inttoptr i32 %x to i32 addrspace(2)*
  ret i32 addrspace(2)* %y
}

; The intermediate integer size is too small, nothing should happen
define i32 addrspace(3)* @test_constant_fold_inttoptr_as_pointer_larger() {
; CHECK-LABEL: @test_constant_fold_inttoptr_as_pointer_larger(
; CHECK-NEXT: ret i32 addrspace(3)* inttoptr (i8 ptrtoint (i32 addrspace(3)* @const_zero_i32_as3 to i8) to i32 addrspace(3)*)
  %x = ptrtoint i32 addrspace(3)* @const_zero_i32_as3 to i8
  %y = inttoptr i8 %x to i32 addrspace(3)*
  ret i32 addrspace(3)* %y
}

define i8 @const_fold_ptrtoint() {
; CHECK-LABEL: @const_fold_ptrtoint(
; CHECK-NEXT: ret i8 4
  ret i8 ptrtoint (i32 addrspace(2)* inttoptr (i4 4 to i32 addrspace(2)*) to i8)
}

; Test that mask happens when the destination pointer is smaller than
; the original
define i8 @const_fold_ptrtoint_mask() {
; CHECK-LABEL: @const_fold_ptrtoint_mask(
; CHECK-NEXT: ret i8 1
  ret i8 ptrtoint (i32 addrspace(3)* inttoptr (i32 257 to i32 addrspace(3)*) to i8)
}

; Address space 0 is too small for the correct mask, should mask with
; 64-bits instead of 32
define i64 @const_fold_ptrtoint_mask_small_as0() {
; CHECK-LABEL: @const_fold_ptrtoint_mask_small_as0(
; CHECK: ret i64 -1
  ret i64 ptrtoint (i32 addrspace(1)* inttoptr (i128 -1 to i32 addrspace(1)*) to i64)
}

define i32 addrspace(3)* @const_inttoptr() {
; CHECK-LABEL: @const_inttoptr(
; CHECK-NEXT: ret i32 addrspace(3)* inttoptr (i16 4 to i32 addrspace(3)*)
  %p = inttoptr i16 4 to i32 addrspace(3)*
  ret i32 addrspace(3)* %p
}

define i16 @const_ptrtoint() {
; CHECK-LABEL: @const_ptrtoint(
; CHECK-NEXT: ret i16 ptrtoint (i32 addrspace(3)* @g to i16)
  %i = ptrtoint i32 addrspace(3)* @g to i16
  ret i16 %i
}

define i16 @const_inttoptr_ptrtoint() {
; CHECK-LABEL: @const_inttoptr_ptrtoint(
; CHECK-NEXT: ret i16 9
  ret i16 ptrtoint (i32 addrspace(3)* inttoptr (i16 9 to i32 addrspace(3)*) to i16)
}

define i1 @constant_fold_cmp_constantexpr_inttoptr() {
; CHECK-LABEL: @constant_fold_cmp_constantexpr_inttoptr(
; CHECK-NEXT: ret i1 true
  %x = icmp eq i32 addrspace(3)* inttoptr (i16 0 to i32 addrspace(3)*), null
  ret i1 %x
}

define i1 @constant_fold_inttoptr_null(i16 %i) {
; CHECK-LABEL: @constant_fold_inttoptr_null(
; CHECK-NEXT: ret i1 false
  %x = icmp eq i32 addrspace(3)* inttoptr (i16 99 to i32 addrspace(3)*), inttoptr (i16 0 to i32 addrspace(3)*)
  ret i1 %x
}

define i1 @constant_fold_ptrtoint_null() {
; CHECK-LABEL: @constant_fold_ptrtoint_null(
; CHECK-NEXT: ret i1 false
  %x = icmp eq i16 ptrtoint (i32 addrspace(3)* @g to i16), ptrtoint (i32 addrspace(3)* null to i16)
  ret i1 %x
}

define i1 @constant_fold_ptrtoint_null_2() {
; CHECK-LABEL: @constant_fold_ptrtoint_null_2(
; CHECK-NEXT: ret i1 false
  %x = icmp eq i16 ptrtoint (i32 addrspace(3)* null to i16), ptrtoint (i32 addrspace(3)* @g to i16)
  ret i1 %x
}

define i1 @constant_fold_ptrtoint() {
; CHECK-LABEL: @constant_fold_ptrtoint(
; CHECK-NEXT: ret i1 true
  %x = icmp eq i16 ptrtoint (i32 addrspace(3)* @g to i16), ptrtoint (i32 addrspace(3)* @g to i16)
  ret i1 %x
}

define i1 @constant_fold_inttoptr() {
; CHECK-LABEL: @constant_fold_inttoptr(
; CHECK-NEXT: ret i1 false
  %x = icmp eq i32 addrspace(3)* inttoptr (i16 99 to i32 addrspace(3)*), inttoptr (i16 27 to i32 addrspace(3)*)
  ret i1 %x
}

@g_float_as3 = addrspace(3) global float zeroinitializer
@g_v4f_as3 = addrspace(3) global <4 x float> zeroinitializer

define float @constant_fold_bitcast_ftoi_load() {
; CHECK-LABEL: @constant_fold_bitcast_ftoi_load(
; CHECK: load float, float addrspace(3)* bitcast (i32 addrspace(3)* @g to float addrspace(3)*), align 4
  %a = load float, float addrspace(3)* bitcast (i32 addrspace(3)* @g to float addrspace(3)*), align 4
  ret float %a
}

define i32 @constant_fold_bitcast_itof_load() {
; CHECK-LABEL: @constant_fold_bitcast_itof_load(
; CHECK: load i32, i32 addrspace(3)* bitcast (float addrspace(3)* @g_float_as3 to i32 addrspace(3)*), align 4
  %a = load i32, i32 addrspace(3)* bitcast (float addrspace(3)* @g_float_as3 to i32 addrspace(3)*), align 4
  ret i32 %a
}

define <4 x float> @constant_fold_bitcast_vector_as() {
; CHECK-LABEL: @constant_fold_bitcast_vector_as(
; CHECK: load <4 x float>, <4 x float> addrspace(3)* @g_v4f_as3, align 16
  %a = load <4 x float>, <4 x float> addrspace(3)* bitcast (<4 x i32> addrspace(3)* bitcast (<4 x float> addrspace(3)* @g_v4f_as3 to <4 x i32> addrspace(3)*) to <4 x float> addrspace(3)*), align 4
  ret <4 x float> %a
}

@i32_array_as3 = addrspace(3) global [10 x i32] zeroinitializer

define i32 @test_cast_gep_small_indices_as() {
; CHECK-LABEL: @test_cast_gep_small_indices_as(
; CHECK: load i32, i32 addrspace(3)* getelementptr inbounds ([10 x i32], [10 x i32] addrspace(3)* @i32_array_as3, i16 0, i16 0), align 16
   %p = getelementptr [10 x i32], [10 x i32] addrspace(3)* @i32_array_as3, i7 0, i7 0
   %x = load i32, i32 addrspace(3)* %p, align 4
   ret i32 %x
}

%struct.foo = type { float, float, [4 x i32], i32 addrspace(3)* }

@constant_fold_global_ptr = addrspace(3) global %struct.foo {
  float 0.0,
  float 0.0,
  [4 x i32] zeroinitializer,
  i32 addrspace(3)* getelementptr ([10 x i32], [10 x i32] addrspace(3)* @i32_array_as3, i64 0, i64 0)
}

define i32 @test_cast_gep_large_indices_as() {
; CHECK-LABEL: @test_cast_gep_large_indices_as(
; CHECK: load i32, i32 addrspace(3)* getelementptr inbounds ([10 x i32], [10 x i32] addrspace(3)* @i32_array_as3, i16 0, i16 0), align 16
   %p = getelementptr [10 x i32], [10 x i32] addrspace(3)* @i32_array_as3, i64 0, i64 0
   %x = load i32, i32 addrspace(3)* %p, align 4
   ret i32 %x
}

define i32 @test_constant_cast_gep_struct_indices_as() {
; CHECK-LABEL: @test_constant_cast_gep_struct_indices_as(
; CHECK: load i32, i32 addrspace(3)* getelementptr inbounds (%struct.foo, %struct.foo addrspace(3)* @constant_fold_global_ptr, i16 0, i32 2, i16 2), align 8
  %x = getelementptr %struct.foo, %struct.foo addrspace(3)* @constant_fold_global_ptr, i18 0, i32 2, i12 2
  %y = load i32, i32 addrspace(3)* %x, align 4
  ret i32 %y
}

@constant_data_as3 = addrspace(3) constant [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5]

define i32 @test_read_data_from_global_as3() {
; CHECK-LABEL: @test_read_data_from_global_as3(
; CHECK-NEXT: ret i32 2
  %x = getelementptr [5 x i32], [5 x i32] addrspace(3)* @constant_data_as3, i32 0, i32 1
  %y = load i32, i32 addrspace(3)* %x, align 4
  ret i32 %y
}

@a = addrspace(1) constant i32 9
@b = addrspace(1) constant i32 23
@c = addrspace(1) constant i32 34
@d = addrspace(1) constant i32 99

@ptr_array = addrspace(2) constant [4 x i32 addrspace(1)*] [ i32 addrspace(1)* @a, i32 addrspace(1)* @b, i32 addrspace(1)* @c, i32 addrspace(1)* @d]
@indirect = addrspace(0) constant i32 addrspace(1)* addrspace(2)* getelementptr inbounds ([4 x i32 addrspace(1)*], [4 x i32 addrspace(1)*] addrspace(2)* @ptr_array, i1 0, i32 2)

define i32 @constant_through_array_as_ptrs() {
; CHECK-LABEL: @constant_through_array_as_ptrs(
; CHECK-NEXT: ret i32 34
  %p = load i32 addrspace(1)* addrspace(2)*, i32 addrspace(1)* addrspace(2)* addrspace(0)* @indirect, align 4
  %a = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(2)* %p, align 4
  %b = load i32, i32 addrspace(1)* %a, align 4
  ret i32 %b
}

@shared_mem = external addrspace(3) global [0 x i8]

define float @canonicalize_addrspacecast(i32 %i) {
; CHECK-LABEL: @canonicalize_addrspacecast
; CHECK-NEXT: getelementptr inbounds float, float* addrspacecast (float addrspace(3)* bitcast ([0 x i8] addrspace(3)* @shared_mem to float addrspace(3)*) to float*), i32 %i
  %p = getelementptr inbounds float, float* addrspacecast ([0 x i8] addrspace(3)* @shared_mem to float*), i32 %i
  %v = load float, float* %p
  ret float %v
}
