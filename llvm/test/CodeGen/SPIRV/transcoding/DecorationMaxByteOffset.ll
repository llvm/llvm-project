; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64v1.1-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.1-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.1 %}
; RUN: llc -O0 -mtriple=spirv32v1.0-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV_1_0
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.0-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.0 %}
; RUN: llc -O0 -mtriple=spirv64v1.0-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV_1_0
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.0-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.0 %}

; CHECK-SPIRV:     OpName %[[#PTR_ID:]] "ptr"
; CHECK-SPIRV:     OpName %[[#PTR2_ID:]] "ptr2"
; CHECK-SPIRV-DAG: OpDecorate %[[#PTR_ID]] MaxByteOffset 12
; CHECK-SPIRV-DAG: OpDecorate %[[#PTR2_ID]] MaxByteOffset 123
; CHECK-SPIRV:     %[[#CHAR_T:]] = OpTypeInt 8 0
; CHECK-SPIRV:     %[[#CHAR_PTR_T:]] = OpTypePointer Workgroup %[[#CHAR_T]]
; CHECK-SPIRV:     %[[#PTR_ID]] = OpFunctionParameter %[[#CHAR_PTR_T]]
; CHECK-SPIRV:     %[[#PTR2_ID]] = OpFunctionParameter %[[#CHAR_PTR_T]]

; CHECK-SPIRV_1_0-NOT: MaxByteOffset
; CHECK-SPIRV_1_0:     OpFunctionParameter

define spir_kernel void @worker(ptr addrspace(3) dereferenceable(12) %ptr) {
entry:
  %ptr.addr = alloca ptr addrspace(3), align 4
  store ptr addrspace(3) %ptr, ptr %ptr.addr, align 4
  ret void
}

define spir_func void @not_a_kernel(ptr addrspace(3) dereferenceable(123) %ptr2) {
entry:
  ret void
}
