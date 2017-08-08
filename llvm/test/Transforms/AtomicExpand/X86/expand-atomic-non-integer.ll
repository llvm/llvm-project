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

define void @pointer_cmpxchg_expand(i8** %ptr, i8* %v) {
; CHECK-LABEL: @pointer_cmpxchg_expand
; CHECK: %1 = bitcast i8** %ptr to i64*
; CHECK: %2 = ptrtoint i8* %v to i64
; CHECK: %3 = cmpxchg i64* %1, i64 0, i64 %2 seq_cst monotonic
; CHECK: %4 = extractvalue { i64, i1 } %3, 0
; CHECK: %5 = extractvalue { i64, i1 } %3, 1
; CHECK: %6 = inttoptr i64 %4 to i8*
; CHECK: %7 = insertvalue { i8*, i1 } undef, i8* %6, 0
; CHECK: %8 = insertvalue { i8*, i1 } %7, i1 %5, 1
  cmpxchg i8** %ptr, i8* null, i8* %v seq_cst monotonic
  ret void
}

define void @pointer_cmpxchg_expand2(i8** %ptr, i8* %v) {
; CHECK-LABEL: @pointer_cmpxchg_expand2
; CHECK: %1 = bitcast i8** %ptr to i64*
; CHECK: %2 = ptrtoint i8* %v to i64
; CHECK: %3 = cmpxchg i64* %1, i64 0, i64 %2 release monotonic
; CHECK: %4 = extractvalue { i64, i1 } %3, 0
; CHECK: %5 = extractvalue { i64, i1 } %3, 1
; CHECK: %6 = inttoptr i64 %4 to i8*
; CHECK: %7 = insertvalue { i8*, i1 } undef, i8* %6, 0
; CHECK: %8 = insertvalue { i8*, i1 } %7, i1 %5, 1
  cmpxchg i8** %ptr, i8* null, i8* %v release monotonic
  ret void
}

define void @pointer_cmpxchg_expand3(i8** %ptr, i8* %v) {
; CHECK-LABEL: @pointer_cmpxchg_expand3
; CHECK: %1 = bitcast i8** %ptr to i64*
; CHECK: %2 = ptrtoint i8* %v to i64
; CHECK: %3 = cmpxchg i64* %1, i64 0, i64 %2 seq_cst seq_cst
; CHECK: %4 = extractvalue { i64, i1 } %3, 0
; CHECK: %5 = extractvalue { i64, i1 } %3, 1
; CHECK: %6 = inttoptr i64 %4 to i8*
; CHECK: %7 = insertvalue { i8*, i1 } undef, i8* %6, 0
; CHECK: %8 = insertvalue { i8*, i1 } %7, i1 %5, 1
  cmpxchg i8** %ptr, i8* null, i8* %v seq_cst seq_cst
  ret void
}

define void @pointer_cmpxchg_expand4(i8** %ptr, i8* %v) {
; CHECK-LABEL: @pointer_cmpxchg_expand4
; CHECK: %1 = bitcast i8** %ptr to i64*
; CHECK: %2 = ptrtoint i8* %v to i64
; CHECK: %3 = cmpxchg weak i64* %1, i64 0, i64 %2 seq_cst seq_cst
; CHECK: %4 = extractvalue { i64, i1 } %3, 0
; CHECK: %5 = extractvalue { i64, i1 } %3, 1
; CHECK: %6 = inttoptr i64 %4 to i8*
; CHECK: %7 = insertvalue { i8*, i1 } undef, i8* %6, 0
; CHECK: %8 = insertvalue { i8*, i1 } %7, i1 %5, 1
  cmpxchg weak i8** %ptr, i8* null, i8* %v seq_cst seq_cst
  ret void
}

define void @pointer_cmpxchg_expand5(i8** %ptr, i8* %v) {
; CHECK-LABEL: @pointer_cmpxchg_expand5
; CHECK: %1 = bitcast i8** %ptr to i64*
; CHECK: %2 = ptrtoint i8* %v to i64
; CHECK: %3 = cmpxchg volatile i64* %1, i64 0, i64 %2 seq_cst seq_cst
; CHECK: %4 = extractvalue { i64, i1 } %3, 0
; CHECK: %5 = extractvalue { i64, i1 } %3, 1
; CHECK: %6 = inttoptr i64 %4 to i8*
; CHECK: %7 = insertvalue { i8*, i1 } undef, i8* %6, 0
; CHECK: %8 = insertvalue { i8*, i1 } %7, i1 %5, 1
  cmpxchg volatile i8** %ptr, i8* null, i8* %v seq_cst seq_cst
  ret void
}

define void @pointer_cmpxchg_expand6(i8 addrspace(2)* addrspace(1)* %ptr, 
                                     i8 addrspace(2)* %v) {
; CHECK-LABEL: @pointer_cmpxchg_expand6
; CHECK: %1 = bitcast i8 addrspace(2)* addrspace(1)* %ptr to i64 addrspace(1)*
; CHECK: %2 = ptrtoint i8 addrspace(2)* %v to i64
; CHECK: %3 = cmpxchg i64 addrspace(1)* %1, i64 0, i64 %2 seq_cst seq_cst
; CHECK: %4 = extractvalue { i64, i1 } %3, 0
; CHECK: %5 = extractvalue { i64, i1 } %3, 1
; CHECK: %6 = inttoptr i64 %4 to i8 addrspace(2)*
; CHECK: %7 = insertvalue { i8 addrspace(2)*, i1 } undef, i8 addrspace(2)* %6, 0
; CHECK: %8 = insertvalue { i8 addrspace(2)*, i1 } %7, i1 %5, 1
  cmpxchg i8 addrspace(2)* addrspace(1)* %ptr, i8 addrspace(2)* null, i8 addrspace(2)* %v seq_cst seq_cst
  ret void
}

