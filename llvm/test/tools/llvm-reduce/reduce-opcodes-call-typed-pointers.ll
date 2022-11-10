; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=opcodes --test FileCheck --test-arg --check-prefix=ALL --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,ALL %s < %t

target datalayout = "A5"

; ALL-LABEL: @call_void_no_args(
; RESULT-NEXT: store volatile i32 0, i32 addrspace(5)* null, align 4
; RESULT-NEXT: ret void
define void @call_void_no_args() {
  call void @void_no_args()
  ret void
}

; ALL-LABEL: @call_load_like_i32(
; RESULT-NEXT: %op = load  volatile i32, i32 addrspace(1)* %ptr, align 4
; RESULT-NEXT: ret i32 %op
define i32 @call_load_like_i32(i32 addrspace(1)* %ptr) {
  %op = call i32 @load_like_i32(i32 addrspace(1)* %ptr)
  ret i32 %op
}

; ALL-LABEL: @call_load_like_ptr_ptr(
; RESULT-NEXT: %op = load volatile i32 addrspace(1)*, i32 addrspace(1)* addrspace(3)* %ptr, align 8
; RESULT-NEXT: ret i32 addrspace(1)* %op
define i32 addrspace(1)* @call_load_like_ptr_ptr(i32 addrspace(1)* addrspace(3)* %ptr) {
  %op = call i32 addrspace(1)* @load_like_ptr_ptr(i32 addrspace(1)* addrspace(3)* %ptr)
  ret i32 addrspace(1)* %op
}

; ALL-LABEL: @call_store_like_i16(
; RESULT-NEXT: store volatile i16 %val, i16 addrspace(1)* %ptr, align 2
; RESULT-NEXT: ret void
define void @call_store_like_i16(i16 %val, i16 addrspace(1)* %ptr) {
  call void @store_like_i16(i16 %val, i16 addrspace(1)* %ptr)
  ret void
}

; ALL-LABEL: @call_load_like_ptr_mismatch(
; RESULT-NEXT: %op = call i32 @load_like_ptr_mismatch(i16 addrspace(1)* %ptr)
; RESULT-NEXT: ret i32 %op
define i32 @call_load_like_ptr_mismatch(i16 addrspace(1)* %ptr) {
  %op = call i32 @load_like_ptr_mismatch(i16 addrspace(1)* %ptr)
  ret i32 %op
}

; ALL-LABEL: @call_store_like_ptr_store(
; RESULT-NEXT: call
; RESULT-NEXT: ret void
define void @call_store_like_ptr_store(i32 addrspace(3)* %ptr.val, i32 addrspace(1)* %ptr) {
  call void @store_like_ptr_store(i32 addrspace(3)* %ptr.val, i32 addrspace(1)* %ptr)
  ret void
}


; ALL-LABEL: @call_store_like_ptr_store_swap(
; RESULT-NEXT: call
; RESULT-NEXT: ret void
define void @call_store_like_ptr_store_swap(i32 addrspace(1)* %ptr, i32 addrspace(3)* %ptr.val) {
  call void @store_like_ptr_store_swap(i32 addrspace(1)* %ptr, i32 addrspace(3)* %ptr.val)
  ret void
}

; ALL-LABEL: @call_store_like_ptr_store_different_element_type(
; RESULT-NEXT: call
; RESULT-NEXT: ret void
define void @call_store_like_ptr_store_different_element_type(i32 addrspace(3)* %ptr.val, i16 addrspace(1)* %ptr) {
  call void @store_like_ptr_store_different_element_type(i32 addrspace(3)* %ptr.val, i16 addrspace(1)* %ptr)
  ret void
}

; ALL-LABEL: @call_store_like_ptr_store_different_element_type_swap(
; RESULT-NEXT: call
; RESULT-NEXT: ret void
define void @call_store_like_ptr_store_different_element_type_swap(i16 addrspace(1)* %ptr, i32 addrspace(3)* %ptr.val) {
  call void @store_like_ptr_store_different_element_type_swap(i16 addrspace(1)* %ptr, i32 addrspace(3)* %ptr.val)
  ret void
}

declare void @void_no_args()
declare i32 addrspace(1)* @load_like_ptr_ptr(i32 addrspace(1)* addrspace(3)*)
declare i32 @load_like_i32(i32 addrspace(1)*)
declare void @store_like_i16(i16 %val, i16 addrspace(1)* %ptr)
declare i32 @load_like_ptr_mismatch(i16 addrspace(1)*)
declare void @store_like_ptr_store(i32 addrspace(3)* %ptr.val, i32 addrspace(1)* %ptr)
declare void @store_like_ptr_store_swap(i32 addrspace(1)* %ptr, i32 addrspace(3)* %ptr.val)
declare void @store_like_ptr_store_different_element_type(i32 addrspace(3)* %ptr.val, i16 addrspace(1)* %ptr)
declare void @store_like_ptr_store_different_element_type_swap(i16 addrspace(1)*, i32 addrspace(3)*)
