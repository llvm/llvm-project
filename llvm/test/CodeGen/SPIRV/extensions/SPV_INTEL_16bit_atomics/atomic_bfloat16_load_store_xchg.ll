; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics,+SPV_KHR_bfloat16 %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: The atomic bfloat16 instruction requires the following SPIR-V extension: SPV_INTEL_16bit_atomics

; CHECK-DAG: Capability BFloat16TypeKHR
; CHECK-DAG: Capability AtomicBFloat16LoadStoreINTEL
; CHECK-DAG: Extension "SPV_KHR_bfloat16"
; CHECK-DAG: Extension "SPV_INTEL_16bit_atomics"
; CHECK-DAG: %[[#TyBF16:]] = OpTypeFloat 16 0
; CHECK-DAG: %[[#TyBF16Ptr:]] = OpTypePointer {{[a-zA-Z]+}} %[[#TyBF16]]
; CHECK-DAG: %[[#TyInt32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Value42:]] = OpConstant %[[#TyBF16]] 16936{{$}}
; CHECK-DAG: %[[#Const0:]] = OpConstantNull %[[#TyBF16]]
; CHECK-DAG: %[[#BF16Ptr:]] = OpVariable %[[#TyBF16Ptr]] CrossWorkgroup %[[#Const0]]
; CHECK-DAG: %[[#ScopeDevice:]] = OpConstant %[[#TyInt32]] 1{{$}}
; CHECK-DAG: %[[#MemSemAcqRel:]] = OpConstant %[[#TyInt32]] 528{{$}}
; CHECK-DAG: %[[#ScopeAllSvmDevices:]] = OpConstantNull %[[#TyInt32]]
; CHECK-DAG: %[[#MemSeqCst:]] = OpConstant %[[#TyInt32]] 16{{$}}

; CHECK: OpAtomicLoad %[[#TyBF16]] %[[#BF16Ptr]] %[[#ScopeDevice]] %[[#MemSemAcqRel]]
; CHECK: OpAtomicStore %[[#BF16Ptr]] %[[#ScopeDevice]] %[[#MemSemAcqRel]] %[[#Value42]]
; CHECK: OpAtomicExchange %[[#TyBF16]] %[[#BF16Ptr]] %[[#ScopeAllSvmDevices]] %[[#MemSeqCst]] %[[#Value42]]


@val = common dso_local local_unnamed_addr addrspace(1) global bfloat 0.000000e+00, align 2

define dso_local spir_func void @test_atomic_bfloat16_load_store_xchg() local_unnamed_addr {
entry:
  %load = call spir_func bfloat @atomic_load(ptr addrspace(1) @val)
  call spir_func void @atomic_store(ptr addrspace(1) @val, bfloat 42.000000e+00)
  %xchg = atomicrmw xchg ptr addrspace(1) @val, bfloat 42.000000e+00 seq_cst
  ret void
}

declare spir_func bfloat @atomic_load(ptr addrspace(1))
declare spir_func void @atomic_store(ptr addrspace(1), bfloat)
