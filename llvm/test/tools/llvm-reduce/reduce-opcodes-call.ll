; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=opcodes --test FileCheck --test-arg --check-prefix=ALL --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,ALL %s < %t

target datalayout = "A5"

declare token @llvm.return.token()
declare void @llvm.uses.token(token)

; ALL-LABEL: @call_token(
; RESULT-NEXT: %token = call token @llvm.return.token()
; RESULT-NEXT: call void @llvm.uses.token(token %token)
; RESULT-NEXT: ret void
define void @call_token() {
  %token = call token @llvm.return.token()
  call void @llvm.uses.token(token %token)
  ret void
}

; ALL-LABEL: @call_void_0_size_arg(
; RESULT-NEXT: store volatile {} %arg, ptr addrspace(5) null, align 1
; RESULT-NEXT: ret void
define void @call_void_0_size_arg({} %arg) {
  call void @void_0_size_arg({} %arg)
  ret void
}

; ALL-LABEL: @call_return_0_size(
; RESULT-NEXT:  %op = load volatile {}, ptr %ptr, align 1
; RESULT-NEXT: ret {} %op
define {} @call_return_0_size(ptr %ptr) {
  %op = call {} @return_0_size(ptr %ptr)
  ret {} %op
}

; ALL-LABEL: define void @call_void_no_args(
; RESULT-NEXT: store volatile i32 0, ptr addrspace(5) null, align 4
; RESULT-NEXT: ret void
define void @call_void_no_args() {
  call void @void_no_args()
  ret void
}

; ALL-LABEL: @call_store_like_i16(
; RESULT-NEXT: store volatile i16 %val, ptr addrspace(1) %ptr, align 2
; RESULT-NEXT: ret void
define void @call_store_like_i16(i16 %val, ptr addrspace(1) %ptr) {
  call void @store_like_i16(i16 %val, ptr addrspace(1) %ptr)
  ret void
}

; ALL-LABEL: @keep_call_store_like_i16(
; ALL-NEXT: call void @store_like_i16(i16 %val, ptr addrspace(1) %ptr)
; ALL-NEXT: ret void
define void @keep_call_store_like_i16(i16 %val, ptr addrspace(1) %ptr) {
  call void @store_like_i16(i16 %val, ptr addrspace(1) %ptr)
  ret void
}

; ALL-LABEL: @call_store_like_i16_swap(
; RESULT-NEXT: store volatile i16 %val, ptr addrspace(1) %ptr
; RESULT-NEXT: ret void
define void @call_store_like_i16_swap(ptr addrspace(1) %ptr, i16 %val) {
  call void @store_like_i16_swap(ptr addrspace(1) %ptr, i16 %val)
  ret void
}

; ALL-LABEL: @call_store_like_i16_extra_arg(
; RESULT-NEXT: call void @store_like_i16_extra_arg(i16 %val, ptr addrspace(1) %ptr, i32 %extra)
; RESULT-NEXT: ret void
define void @call_store_like_i16_extra_arg(i16 %val, ptr addrspace(1) %ptr, i32 %extra) {
  call void @store_like_i16_extra_arg(i16 %val, ptr addrspace(1) %ptr, i32 %extra)
  ret void
}

; ALL-LABEL: @call_store_like_i16_extra_ptr_arg(
; RESULT-NEXT: call void @store_like_i16_extra_ptr_arg(i16 %val, ptr addrspace(1) %ptr, ptr addrspace(1) %extra)
; RESULT-NEXT: ret void
define void @call_store_like_i16_extra_ptr_arg(i16 %val, ptr addrspace(1) %ptr, ptr addrspace(1) %extra) {
  call void @store_like_i16_extra_ptr_arg(i16 %val, ptr addrspace(1) %ptr, ptr addrspace(1) %extra)
  ret void
}

; ALL-LABEL: @call_store_like_ptr_store(
; RESULT-NEXT: store volatile ptr addrspace(1) %ptr, ptr addrspace(3) %ptr.val, align 8
; RESULT-NEXT: ret void
define void @call_store_like_ptr_store(ptr addrspace(3) %ptr.val, ptr addrspace(1) %ptr) {
  call void @store_like_ptr_store(ptr addrspace(3) %ptr.val, ptr addrspace(1) %ptr)
  ret void
}

; ALL-LABEL: @call_store_like_ptr_store_swap(
; RESULT-NEXT: store volatile ptr addrspace(3) %ptr.val, ptr addrspace(1) %ptr, align 8
; RESULT-NEXT: ret void
define void @call_store_like_ptr_store_swap(ptr addrspace(1) %ptr, ptr addrspace(3) %ptr.val) {
  call void @store_like_ptr_store_swap(ptr addrspace(1) %ptr, ptr addrspace(3) %ptr.val)
  ret void
}

; ALL-LABEL: @call_store_like_ptr_store_different_element_type(
; RESULT-NEXT: store volatile ptr addrspace(1) %ptr, ptr addrspace(3) %ptr.val, align 8
; RESULT-NEXT: ret void
define void @call_store_like_ptr_store_different_element_type(ptr addrspace(3) %ptr.val, ptr addrspace(1) %ptr) {
  call void @store_like_ptr_store_different_element_type(ptr addrspace(3) %ptr.val, ptr addrspace(1) %ptr)
  ret void
}

; ALL-LABEL: @call_store_like_ptr_store_different_element_type_swap(
; RESULT-NEXT: store volatile ptr addrspace(3) %ptr.val, ptr addrspace(1) %ptr, align 8
; RESULT-NEXT: ret void
define void @call_store_like_ptr_store_different_element_type_swap(ptr addrspace(1) %ptr, ptr addrspace(3) %ptr.val) {
  call void @store_like_ptr_store_different_element_type_swap(ptr addrspace(1) %ptr, ptr addrspace(3) %ptr.val)
  ret void
}

; ALL-LABEL: @call_load_like_i32(
; RESULT-NEXT: %op = load volatile i32, ptr addrspace(1) %ptr, align 4
; RESULT-NEXT: ret i32 %op
define i32 @call_load_like_i32(ptr addrspace(1) %ptr) {
  %op = call i32 @load_like_i32(ptr addrspace(1) %ptr)
  ret i32 %op
}

; ALL-LABEL: @keep_call_load_like_i32(
; ALL-NEXT: %op = call i32 @load_like_i32(ptr addrspace(1) %ptr)
; ALL-NEXT: ret i32 %op
define i32 @keep_call_load_like_i32(ptr addrspace(1) %ptr) {
  %op = call i32 @load_like_i32(ptr addrspace(1) %ptr)
  ret i32 %op
}

; ALL-LABEL: @call_load_like_i32_extra_arg(
; RESULT-NEXT: %op = call i32 @load_like_i32_extra_arg(ptr addrspace(1) %ptr, i32 %extra)
; RESULT-NEXT: ret i32 %op
define i32 @call_load_like_i32_extra_arg(ptr addrspace(1) %ptr, i32 %extra) {
  %op = call i32 @load_like_i32_extra_arg(ptr addrspace(1) %ptr, i32 %extra)
  ret i32 %op
}

; ALL-LABEL: @call_load_like_ptr_mismatch(
; RESULT-NEXT: %op = load volatile i32, ptr addrspace(1) %ptr, align 4
; RESULT-NEXT: ret i32 %op
define i32 @call_load_like_ptr_mismatch(ptr addrspace(1) %ptr) {
  %op = call i32 @load_like_ptr_mismatch(ptr addrspace(1) %ptr)
  ret i32 %op
}

; ALL-LABEL: @call_load_like_skip_arg(
; RESULT-NEXT: %op = load volatile i32, ptr addrspace(1) %ptr, align 4
; RESULT-NEXT: ret i32 %op
define i32 @call_load_like_skip_arg(float, ptr addrspace(1) %ptr) {
  %op = call i32 @load_like_skip_arg(float poison, ptr addrspace(1) %ptr)
  ret i32 %op
}

; ALL-LABEL: @call_fp_scalar_noargs(
; RESULT-NEXT: %op = load volatile float, ptr addrspace(5) null, align 4
; RESULT-NEXT: ret float %op
define float @call_fp_scalar_noargs() {
  %op = call nsz float @fp_scalar_noargs()
  ret float %op
}

; ALL-LABEL: @call_fp_vector_noargs(
; RESULT-NEXT: %op = load volatile <2 x half>, ptr addrspace(5) null, align 4
; RESULT-NEXT: ret <2 x half> %op
define <2 x half> @call_fp_vector_noargs() {
  %op = call nsz <2 x half> @fp_vector_noargs()
  ret <2 x half> %op
}

; ALL-LABEL: @call_unary_fp_scalar(
; RESULT-NEXT: %op = fneg nsz float %a
; RESULT-NEXT: ret float %op
define float @call_unary_fp_scalar(float %a) {
  %op = call nsz float @unary_fp_scalar(float %a)
  ret float %op
}

; ALL-LABEL: @call_unary_fp_vector(
; RESULT-NEXT: %op = fneg nsz <2 x half> %a
; RESULT-NEXT: ret <2 x half> %op
define <2 x half> @call_unary_fp_vector(<2 x half> %a) {
  %op = call nsz <2 x half> @unary_fp_vector(<2 x half> %a)
  ret <2 x half> %op
}

; ALL-LABEL: @ignore_undef_args_unary_fp(
; RESULT-NEXT: %op = fneg nnan float %a
; RESULT-NEXT: ret float %op
define float @ignore_undef_args_unary_fp(float %a) {
  %op = call nnan float @func_i32_f32_i32(i32 poison, float %a, i32 poison)
  ret float %op
}

; ALL-LABEL: @call_binary_fp_scalar(
; RESULT: %op = fmul afn float %a, %b
; RESULT-NEXT: ret float %op
define float @call_binary_fp_scalar(float %a, float %b) {
  %op = call afn float @binary_fp_scalar(float %a, float %b)
  ret float %op
}

; ALL-LABEL: @call_binary_fp_vector(
; RESULT-NEXT: %op = fmul afn <2 x half> %a, %b
; RESULT-NEXT: ret <2 x half> %op
define <2 x half> @call_binary_fp_vector(<2 x half> %a, <2 x half> %b) {
  %op = call afn <2 x half> @binary_fp_vector(<2 x half> %a, <2 x half> %b)
  ret <2 x half> %op
}

; ALL-LABEL: @call_ternary_fp_scalar(
; RESULT-NEXT: %op = call afn float @llvm.fma.f32(float %a, float %b, float %c)
; RESULT-NEXT: ret float %op
define float @call_ternary_fp_scalar(float %a, float %b, float %c) {
  %op = call afn float @ternary_fp_scalar(float %a, float %b, float %c)
  ret float %op
}

; ALL-LABEL: @call_ternary_fp_vector(
; RESULT-NEXT: %op = call afn <2 x half> @llvm.fma.v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c)
; RESULT-NEXT: ret <2 x half> %op
define <2 x half> @call_ternary_fp_vector(<2 x half> %a, <2 x half> %b, <2 x half> %c) {
  %op = call afn <2 x half> @ternary_fp_vector(<2 x half> %a, <2 x half> %b, <2 x half> %c)
  ret <2 x half> %op
}

; ALL-LABEL: @call_unary_int_scalar(
; RESULT-NEXT: %op = call i32 @llvm.bswap.i32(i32 %a)
; RESULT-NEXT: ret i32 %op
define i32 @call_unary_int_scalar(i32 %a) {
  %op = call i32 @unary_int_scalar(i32 %a)
  ret i32 %op
}

; ALL-LABEL: @call_unary_int_vector(
; RESULT-NEXT: %op = call <2 x i16> @llvm.bswap.v2i16(<2 x i16> %a)
; RESULT-NEXT: ret <2 x i16> %op
define <2 x i16> @call_unary_int_vector(<2 x i16> %a) {
  %op = call <2 x i16> @unary_int_vector(<2 x i16> %a)
  ret <2 x i16> %op
}

; ALL-LABEL: @call_binary_int_scalar(
; RESULT-NEXT: %op = and i32 %a, %b
; RESULT-NEXT: ret i32 %op
define i32 @call_binary_int_scalar(i32 %a, i32 %b) {
  %op = call i32 @binary_int_scalar(i32 %a, i32 %b)
  ret i32 %op
}

; ALL-LABEL: @call_binary_int_vector(
; RESULT-NEXT: %op = and <2 x i16> %a, %b
; RESULT-NEXT: ret <2 x i16> %op
define <2 x i16> @call_binary_int_vector(<2 x i16> %a, <2 x i16> %b) {
  %op = call <2 x i16> @binary_int_vector(<2 x i16> %a, <2 x i16> %b)
  ret <2 x i16> %op
}

; ALL-LABEL: @call_ternary_int_scalar(
; RESULT-NEXT: %op = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
; RESULT-NEXT: ret i32 %op
define i32 @call_ternary_int_scalar(i32 %a, i32 %b, i32 %c) {
  %op = call i32 @ternary_int_scalar(i32 %a, i32 %b, i32 %c)
  ret i32 %op
}

; ALL-LABEL: @call_ternary_int_vector(
; RESULT-NEXT: %op = call <2 x i16> @llvm.fshl.v2i16(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c)
; RESULT-NEXT: ret <2 x i16> %op
define <2 x i16> @call_ternary_int_vector(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c) {
  %op = call <2 x i16> @ternary_int_vector(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c)
  ret <2 x i16> %op
}

; ALL-LABEL: @call_quaternary_int_scalar(
; RESULT-NEXT: %op = call i32 @quaternary_int_scalar(i32 %a, i32 %b, i32 %c, i32 %d)
; RESULT-NEXT: ret i32 %op
define i32 @call_quaternary_int_scalar(i32 %a, i32 %b, i32 %c, i32 %d) {
  %op = call i32 @quaternary_int_scalar(i32 %a, i32 %b, i32 %c, i32 %d)
  ret i32 %op
}

; ALL-LABEL: @call_quaternary_int_vector(
; RESULT-NEXT: %op = call <2 x i16> @quaternary_int_vector(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c, <2 x i16> %d)
; RESULT-NEXT: ret <2 x i16> %op
define <2 x i16> @call_quaternary_int_vector(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c, <2 x i16> %d) {
  %op = call <2 x i16> @quaternary_int_vector(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c, <2 x i16> %d)
  ret <2 x i16> %op
}

declare void @void_0_size_arg({})
declare {} @return_0_size(ptr)
declare void @void_no_args()
declare void @store_like_i16(i16, ptr addrspace(1))
declare void @store_like_i16_swap(ptr addrspace(1), i16)
declare void @store_like_i16_extra_arg(i16, ptr addrspace(1), i32)
declare void @store_like_i16_extra_ptr_arg(i16, ptr addrspace(1), ptr addrspace(1))
declare void @store_like_ptr_store(ptr addrspace(3), ptr addrspace(1))
declare void @store_like_ptr_store_swap(ptr addrspace(1), ptr addrspace(3))
declare void @store_like_ptr_store_different_element_type(ptr addrspace(3), ptr addrspace(1))
declare void @store_like_ptr_store_different_element_type_swap(ptr addrspace(1), ptr addrspace(3))
declare i32 @load_like_i32(ptr addrspace(1))

declare i32 @load_like_i32_extra_arg(ptr addrspace(1), i32)

declare i32 @load_like_ptr_mismatch(ptr addrspace(1))
declare i32 @load_like_skip_arg(float, ptr addrspace(1))

declare float @fp_scalar_noargs()
declare i32 @int_scalar_noargs()

declare <2 x half> @fp_vector_noargs()
declare <2 x i16> @int_vector_noargs()

declare float @unary_fp_scalar(float)
declare <2 x half> @unary_fp_vector(<2 x half>)
declare float @func_i32_f32_i32(i32, float, i32)

declare float @binary_fp_scalar(float, float)
declare <2 x half> @binary_fp_vector(<2 x half>, <2 x half>)

declare float @ternary_fp_scalar(float, float, float)
declare <2 x half> @ternary_fp_vector(<2 x half>, <2 x half>, <2 x half>)

declare float @quaternary_fp_scalar(float, float, float, float)
declare <2 x half> @quaternary_fp_vector(<2 x half>, <2 x half>, <2 x half>, <2 x half>)

declare i32 @unary_int_scalar(i32)
declare <2 x i16> @unary_int_vector(<2 x i16>)
declare i32 @binary_int_scalar(i32, i32)
declare <2 x i16> @binary_int_vector(<2 x i16>, <2 x i16>)
declare i32 @ternary_int_scalar(i32, i32, i32)
declare <2 x i16> @ternary_int_vector(<2 x i16>, <2 x i16>, <2 x i16>)
declare i32 @quaternary_int_scalar(i32, i32, i32, i32)
declare <2 x i16> @quaternary_int_vector(<2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>)
