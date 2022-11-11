; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:     OpName %[[#PTR_ID:]] "ptr"
; CHECK-SPIRV:     OpName %[[#PTR2_ID:]] "ptr2"
; CHECK-SPIRV-DAG: OpDecorate %[[#PTR_ID]] MaxByteOffset 12
; CHECK-SPIRV-DAG: OpDecorate %[[#PTR2_ID]] MaxByteOffset 123
; CHECK-SPIRV:     %[[#CHAR_T:]] = OpTypeInt 8 0
; CHECK-SPIRV:     %[[#CHAR_PTR_T:]] = OpTypePointer Workgroup %[[#CHAR_T]]
; CHECK-SPIRV:     %[[#PTR_ID]] = OpFunctionParameter %[[#CHAR_PTR_T]]
; CHECK-SPIRV:     %[[#PTR2_ID]] = OpFunctionParameter %[[#CHAR_PTR_T]]

define spir_kernel void @worker(i8 addrspace(3)* dereferenceable(12) %ptr) {
entry:
  %ptr.addr = alloca i8 addrspace(3)*, align 4
  store i8 addrspace(3)* %ptr, i8 addrspace(3)** %ptr.addr, align 4
  ret void
}

define spir_func void @not_a_kernel(i8 addrspace(3)* dereferenceable(123) %ptr2) {
entry:
  ret void
}
