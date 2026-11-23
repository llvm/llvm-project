; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics %s -o - | FileCheck %s
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics %s -o - -filetype=obj | spirv-val %}

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
; CHECK-DAG: %[[#ScopeAllSvmDevices:]] = OpConstantNull %[[#TyInt32]]
; CHECK-DAG: %[[#MemSem528:]] = OpConstant %[[#TyInt32]] 528{{$}}

; CHECK-DAG: %[[#Val:]] = OpVariable %[[#TyInt16Ptr]] CrossWorkgroup %[[#Const0]]

; CHECK: OpAtomicLoad %[[#TyInt16]] %[[#Val]] %[[#ScopeAllSvmDevices]] %[[#MemSem528]]
; CHECK: OpAtomicStore %[[#Val]] %[[#ScopeAllSvmDevices]] %[[#MemSem528]] %[[#Const42]]
; CHECK: OpAtomicExchange %[[#TyInt16]] %[[#Val]] %[[#ScopeAllSvmDevices]] %[[#MemSem528]] %[[#Const42]]
; CHECK: OpAtomicCompareExchange %[[#TyInt16]] %[[#]] %[[#ScopeAllSvmDevices]] %[[#MemSem528]] %[[#MemSem528]] %[[#Const42]] %[[#Const1]]


@val = private addrspace(1) global i16 0

define spir_func void @test_atomic_int16_basic() {
entry:
  %load = load atomic i16, ptr addrspace(1) @val seq_cst, align 2
  store atomic i16 42, ptr addrspace(1) @val seq_cst, align 2
  %xchg = atomicrmw xchg ptr addrspace(1) @val, i16 42 seq_cst
  %cmpxchg = cmpxchg ptr addrspace(1) @val, i16 1, i16 42 seq_cst seq_cst
  ret void
}
