; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics,+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics,+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: The atomic bfloat16 instruction requires the following SPIR-V extension: SPV_INTEL_16bit_atomics

; CHECK-DAG: Capability BFloat16TypeKHR
; CHECK-DAG: Capability AtomicBFloat16LoadStoreINTEL
; CHECK-DAG: Extension "SPV_KHR_bfloat16"
; CHECK-DAG: Extension "SPV_INTEL_16bit_atomics"
; CHECK-DAG: %[[#TyBF16:]] = OpTypeFloat 16 0
; CHECK-DAG: %[[#TyBF16Ptr:]] = OpTypePointer CrossWorkgroup %[[#TyBF16]]
; CHECK-DAG: %[[#TyInt32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Const42:]] = OpConstant %[[#TyBF16]] 16936{{$}}
; CHECK-DAG: %[[#Const0:]] = OpConstantNull %[[#TyBF16]]
; CHECK-DAG: %[[#ScopeDevice:]] = OpConstant %[[#TyInt32]] 1{{$}}
; CHECK-DAG: %[[#ScopeAllSvmDevices:]] = OpConstantNull %[[#TyInt32]]
; CHECK-DAG: %[[#MemSem528:]] = OpConstant %[[#TyInt32]] 528{{$}}

; CHECK-DAG: %[[#Val:]] = OpVariable %[[#TyBF16Ptr]] CrossWorkgroup %[[#Const0]]

; CHECK: OpAtomicLoad %[[#TyBF16]] %[[#Val]] %[[#ScopeDevice]] %[[#MemSem528]]
; CHECK: OpAtomicStore %[[#Val]] %[[#ScopeDevice]] %[[#MemSem528]] %[[#Const42]]
; CHECK: OpAtomicExchange %[[#TyBF16]] %[[#Val]] %[[#ScopeAllSvmDevices]] %[[#MemSem528]] %[[#Const42]]


@val = private addrspace(1) global bfloat 0.000000e+00

define spir_func void @test_atomic_bfloat16_load_store_xchg() {
entry:
; TODO: 'load atomic'/'store atomic' LLVM instructions are not yet lowered to
; OpAtomicLoad/OpAtomicStore by the SPIRV backend; use OCL builtins instead.
; TODO: test Vulkan path as well once we support atomic load/store.
  %load = call spir_func bfloat @_Z11atomic_loadPU3AS1VU7_AtomicDF16b(ptr addrspace(1) @val)
  call spir_func void @_Z12atomic_storePU3AS1VU7_AtomicDF16bDF16b(ptr addrspace(1) @val, bfloat 42.000000e+00)
  %xchg = atomicrmw xchg ptr addrspace(1) @val, bfloat 42.000000e+00 seq_cst
  ret void
}

declare spir_func bfloat @_Z11atomic_loadPU3AS1VU7_AtomicDF16b(ptr addrspace(1))
declare spir_func void @_Z12atomic_storePU3AS1VU7_AtomicDF16bDF16b(ptr addrspace(1), bfloat)
