; RUN: llc -O0 -mtriple=spirv64-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-- %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-- %s -o - -filetype=obj | spirv-val %}

;; Check that 'store atomic' LLVM IR instructions are lowered.
;; NOTE: The current lowering is incorrect: 'store atomic' should produce
;; OpAtomicStore but currently produces OpStore, silently dropping the atomic
;; ordering. This test documents the broken behaviour so it can be fixed.

; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int32Vec:]] = OpTypeVector %[[#Int32]] 2

define void @store_i32_unordered(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 4
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr unordered, align 4
  ret void
}

define void @store_i32_monotonic(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 4
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr monotonic, align 4
  ret void
}

define void @store_i32_release(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 4
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr release, align 4
  ret void
}

define void @store_i32_seq_cst(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 4
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr seq_cst, align 4
  ret void
}

; -- test with different syncscopes 

define void @store_i32_release_singlethread(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 4
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr syncscope("singlethread") release, align 4
  ret void
}

define void @store_i32_release_subgroup(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 4
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr syncscope("subgroup") release, align 4
  ret void
}

define void @store_i32_release_workgroup(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 4
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr syncscope("workgroup") release, align 4
  ret void
}

define void @store_i32_release_device(ptr addrspace(1) %ptr, i32 %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 4
; CHECK:       OpReturn
  store atomic i32 %val, ptr addrspace(1) %ptr syncscope("device") release, align 4
  ret void
}

; -- test with a different scalar type

define void @store_float_release(ptr addrspace(1) %ptr, float %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Float]]
; CHECK:       %[[#cast:]] = OpBitcast %[[#Int32]] %[[#val]]
; CHECK:       OpStore %[[#ptr]] %[[#cast]] Aligned 8
; CHECK:       OpReturn
  store atomic float %val, ptr addrspace(1) %ptr release, align 8
  ret void
}

; -- test with a vector type

define void @store_vector_release(ptr addrspace(1) %ptr, <2 x i32> %val) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#val:]] = OpFunctionParameter %[[#Int32Vec]]
; CHECK:       OpStore %[[#ptr]] %[[#val]] Aligned 8
; CHECK:       OpReturn
  store atomic <2 x i32> %val, ptr addrspace(1) %ptr release, align 8
  ret void
}
