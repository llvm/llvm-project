; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16,+SPV_NV_shader_atomic_fp16_vector,+SPV_INTEL_bfloat16_arithmetic %s -o - 2>&1 | FileCheck %s --check-prefix=NOAMD

; SPV_NV_shader_atomic_fp16_vector extension is meant for fp16,
; but the bf16 type uses the same vector atomic so the AMD backend can lower it
; to a packed bf16 atomic.

; Anywhere else the bf16 element type is not allowed on the vector atomic, so the
; same module is rejected.

; NOAMD: cannot be a bfloat16 scalar

; CHECK-DAG: Capability BFloat16TypeKHR
; CHECK-DAG: Capability AtomicFloat16VectorNV
; CHECK-DAG: Extension "SPV_KHR_bfloat16"
; CHECK-DAG: Extension "SPV_NV_shader_atomic_fp16_vector"
; CHECK-DAG: %[[#BF16:]] = OpTypeFloat 16 0
; CHECK-DAG: %[[#V2:]] = OpTypeVector %[[#BF16]] 2
; CHECK-DAG: %[[#V4:]] = OpTypeVector %[[#BF16]] 4

; CHECK: OpFunction %[[#V2]] None %[[#]]
; CHECK: OpAtomicFAddEXT %[[#V2]]
define <2 x bfloat> @test_fadd_v2(ptr addrspace(1) %ptr, <2 x bfloat> %val) {
  %r = atomicrmw fadd ptr addrspace(1) %ptr, <2 x bfloat> %val seq_cst
  ret <2 x bfloat> %r
}

; CHECK: OpFunction %[[#V2]] None %[[#]]
; CHECK: %[[#Neg:]] = OpFNegate %[[#V2]]
; CHECK: OpAtomicFAddEXT %[[#V2]] %[[#]] %[[#]] %[[#]] %[[#Neg]]
define <2 x bfloat> @test_fsub_v2(ptr addrspace(1) %ptr, <2 x bfloat> %val) {
  %r = atomicrmw fsub ptr addrspace(1) %ptr, <2 x bfloat> %val seq_cst
  ret <2 x bfloat> %r
}

; CHECK: OpFunction %[[#V2]] None %[[#]]
; CHECK: OpAtomicFMinEXT %[[#V2]]
define <2 x bfloat> @test_fmin_v2(ptr addrspace(1) %ptr, <2 x bfloat> %val) {
  %r = atomicrmw fmin ptr addrspace(1) %ptr, <2 x bfloat> %val seq_cst
  ret <2 x bfloat> %r
}

; CHECK: OpFunction %[[#V2]] None %[[#]]
; CHECK: OpAtomicFMaxEXT %[[#V2]]
define <2 x bfloat> @test_fmax_v2(ptr addrspace(1) %ptr, <2 x bfloat> %val) {
  %r = atomicrmw fmax ptr addrspace(1) %ptr, <2 x bfloat> %val seq_cst
  ret <2 x bfloat> %r
}

; CHECK: OpFunction %[[#V4]] None %[[#]]
; CHECK: OpAtomicFAddEXT %[[#V4]]
define <4 x bfloat> @test_fadd_v4(ptr addrspace(1) %ptr, <4 x bfloat> %val) {
  %r = atomicrmw fadd ptr addrspace(1) %ptr, <4 x bfloat> %val seq_cst
  ret <4 x bfloat> %r
}
