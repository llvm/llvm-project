; RUN: llc -O0 -mtriple=spirv64-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-- %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-- %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-- %s -o - -filetype=obj | spirv-val %}

;; Check that 'load atomic' LLVM IR instructions are lowered.
;; NOTE: The current lowering is incorrect: 'load atomic' should produce
;; OpAtomicLoad but currently produces OpLoad, silently dropping the atomic
;; ordering. This test documents the broken behaviour so it can be fixed.

; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Int32Vec:]] = OpTypeVector %[[#Int32]] 2

define i32 @load_i32_unordered(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 4
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr unordered, align 4
  ret i32 %val
}

define i32 @load_i32_monotonic(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 4
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr monotonic, align 4
  ret i32 %val
}

define i32 @load_i32_acquire(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 4
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr acquire, align 4
  ret i32 %val
}

define i32 @load_i32_seq_cst(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 4
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr seq_cst, align 4
  ret i32 %val
}

; -- test with different syncscopes 

define i32 @load_i32_acquire_singlethread(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 4
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr syncscope("singlethread") acquire, align 4
  ret i32 %val
}

define i32 @load_i32_acquire_subgroup(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 4
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr syncscope("subgroup") acquire, align 4
  ret i32 %val
}

define i32 @load_i32_acquire_workgroup(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 4
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr syncscope("workgroup") acquire, align 4
  ret i32 %val
}

define i32 @load_i32_acquire_device(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 4
; CHECK:       OpReturnValue
  %val = load atomic i32, ptr addrspace(1) %ptr syncscope("device") acquire, align 4
  ret i32 %val
}

; -- test with a different scalar type

define float @load_float_acquire(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#load:]] = OpLoad %[[#Int32]] %[[#ptr]] Aligned 8
; CHECK:       %[[#val:]] = OpBitcast %[[#Float]] %[[#load]]
; CHECK:       OpReturnValue %[[#val]]
  %val = load atomic float, ptr addrspace(1) %ptr acquire, align 8
  ret float %val
}

; -- test with a vector type

define <2 x i32> @load_vector_acquire(ptr addrspace(1) %ptr) {
; CHECK-LABEL: OpFunction %[[#]]
; CHECK:       %[[#ptr:]] = OpFunctionParameter %[[#]]
; CHECK:       %[[#]] = OpLoad %[[#Int32Vec]] %[[#ptr]] Aligned 8
; CHECK:       OpReturnValue
  %val = load atomic <2 x i32>, ptr addrspace(1) %ptr acquire, align 8
  ret <2 x i32> %val
}
