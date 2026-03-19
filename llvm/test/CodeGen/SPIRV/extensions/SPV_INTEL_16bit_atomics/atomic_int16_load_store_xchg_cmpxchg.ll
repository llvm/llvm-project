; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: 16-bit integer atomic operations require the following SPIR-V extension: SPV_INTEL_16bit_atomics

; CHECK-DAG: Capability Int16
; CHECK-DAG: Capability AtomicInt16CompareExchangeINTEL
; CHECK-DAG: Extension "SPV_INTEL_16bit_atomics"
; CHECK-DAG: %[[#TyInt16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#TyInt16Ptr:]] = OpTypePointer CrossWorkgroup %[[#TyInt16]]
; CHECK-DAG: %[[#TyInt32:]] = OpTypeInt 32 0

; CHECK-DAG: %[[#Const0:]] = OpConstantNull %[[#TyInt16]]
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#TyInt16]] 1{{$}}
; CHECK-DAG: %[[#Const42:]] = OpConstant %[[#TyInt16]] 42{{$}}
; CHECK-DAG: %[[#ScopeDevice:]] = OpConstant %[[#TyInt32]] 1{{$}}
; CHECK-DAG: %[[#ScopeAllSvmDevices:]] = OpConstantNull %[[#TyInt32]]
; CHECK-DAG: %[[#MemSemAcqRel:]] = OpConstant %[[#TyInt32]] 528{{$}}
; CHECK-DAG: %[[#MemSeqCst:]] = OpConstant %[[#TyInt32]] 16{{$}}

; CHECK-DAG: %[[#Val:]] = OpVariable %[[#TyInt16Ptr]] CrossWorkgroup %[[#Const0]]

; CHECK: OpAtomicLoad %[[#TyInt16]] %[[#Val]] %[[#ScopeDevice]] %[[#MemSemAcqRel]]
; CHECK: OpAtomicStore %[[#Val]] %[[#ScopeDevice]] %[[#MemSemAcqRel]] %[[#Const42]]
; CHECK: OpAtomicExchange %[[#TyInt16]] %[[#Val]] %[[#ScopeAllSvmDevices]] %[[#MemSeqCst]] %[[#Const42]]
; CHECK: OpAtomicCompareExchange %[[#TyInt16]] %[[#]] %[[#ScopeAllSvmDevices]] %[[#MemSeqCst]] %[[#MemSeqCst]] %[[#Const42]] %[[#Const1]]


@val = private addrspace(1) global i16 0

define spir_func void @test_atomic_int16_basic() {
entry:
; TODO: 'load atomic'/'store atomic' LLVM instructions are not yet lowered to
; OpAtomicLoad/OpAtomicStore by the SPIRV backend; use OCL builtins instead.
  %load = call spir_func i16 @_Z11atomic_loadPU3AS1VU7_Atomics(ptr addrspace(1) @val)
  call spir_func void @_Z12atomic_storePU3AS1VU7_Atomicss(ptr addrspace(1) @val, i16 42)
  %xchg = atomicrmw xchg ptr addrspace(1) @val, i16 42 seq_cst
  %cmpxchg = cmpxchg ptr addrspace(1) @val, i16 1, i16 42 seq_cst seq_cst
  ret void
}

declare spir_func i16 @_Z11atomic_loadPU3AS1VU7_Atomics(ptr addrspace(1))
declare spir_func void @_Z12atomic_storePU3AS1VU7_Atomicss(ptr addrspace(1), i16)
