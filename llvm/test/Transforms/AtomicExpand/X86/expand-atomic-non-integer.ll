; RUN: opt -S %s -atomic-expand -mtriple=x86_64-linux-gnu | FileCheck %s

; This file tests the functions `llvm::convertAtomicLoadToIntegerType` and
; `llvm::convertAtomicStoreToIntegerType`. If X86 stops using this 
; functionality, please move this test to a target which still is.

define float @float_load_expand(float* %ptr) {
; CHECK-LABEL: @float_load_expand
; CHECK: %1 = bitcast float* %ptr to i32*
; CHECK: %2 = load atomic i32, i32* %1 unordered, align 4
; CHECK: %3 = bitcast i32 %2 to float
; CHECK: ret float %3
  %res = load atomic float, float* %ptr unordered, align 4
  ret float %res
}

define float @float_load_expand_seq_cst(float* %ptr) {
; CHECK-LABEL: @float_load_expand_seq_cst
; CHECK: %1 = bitcast float* %ptr to i32*
; CHECK: %2 = load atomic i32, i32* %1 seq_cst, align 4
; CHECK: %3 = bitcast i32 %2 to float
; CHECK: ret float %3
  %res = load atomic float, float* %ptr seq_cst, align 4
  ret float %res
}

define float @float_load_expand_vol(float* %ptr) {
; CHECK-LABEL: @float_load_expand_vol
; CHECK: %1 = bitcast float* %ptr to i32*
; CHECK: %2 = load atomic volatile i32, i32* %1 unordered, align 4
; CHECK: %3 = bitcast i32 %2 to float
; CHECK: ret float %3
  %res = load atomic volatile float, float* %ptr unordered, align 4
  ret float %res
}

define float @float_load_expand_addr1(float addrspace(1)* %ptr) {
; CHECK-LABEL: @float_load_expand_addr1
; CHECK: %1 = bitcast float addrspace(1)* %ptr to i32 addrspace(1)*
; CHECK: %2 = load atomic i32, i32 addrspace(1)* %1 unordered, align 4
; CHECK: %3 = bitcast i32 %2 to float
; CHECK: ret float %3
  %res = load atomic float, float addrspace(1)* %ptr unordered, align 4
  ret float %res
}

define void @float_store_expand(float* %ptr, float %v) {
; CHECK-LABEL: @float_store_expand
; CHECK: %1 = bitcast float %v to i32
; CHECK: %2 = bitcast float* %ptr to i32*
; CHECK: store atomic i32 %1, i32* %2 unordered, align 4
  store atomic float %v, float* %ptr unordered, align 4
  ret void
}

define void @float_store_expand_seq_cst(float* %ptr, float %v) {
; CHECK-LABEL: @float_store_expand_seq_cst
; CHECK: %1 = bitcast float %v to i32
; CHECK: %2 = bitcast float* %ptr to i32*
; CHECK: store atomic i32 %1, i32* %2 seq_cst, align 4
  store atomic float %v, float* %ptr seq_cst, align 4
  ret void
}

define void @float_store_expand_vol(float* %ptr, float %v) {
; CHECK-LABEL: @float_store_expand_vol
; CHECK: %1 = bitcast float %v to i32
; CHECK: %2 = bitcast float* %ptr to i32*
; CHECK: store atomic volatile i32 %1, i32* %2 unordered, align 4
  store atomic volatile float %v, float* %ptr unordered, align 4
  ret void
}

define void @float_store_expand_addr1(float addrspace(1)* %ptr, float %v) {
; CHECK-LABEL: @float_store_expand_addr1
; CHECK: %1 = bitcast float %v to i32
; CHECK: %2 = bitcast float addrspace(1)* %ptr to i32 addrspace(1)*
; CHECK: store atomic i32 %1, i32 addrspace(1)* %2 unordered, align 4
  store atomic float %v, float addrspace(1)* %ptr unordered, align 4
  ret void
}

