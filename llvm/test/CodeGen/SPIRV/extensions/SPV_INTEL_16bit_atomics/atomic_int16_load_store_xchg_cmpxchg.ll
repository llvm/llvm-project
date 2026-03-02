; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: 16-bit integer atomic operations require the following SPIR-V extension: SPV_INTEL_16bit_atomics

; CHECK-DAG: Capability Int16
; CHECK-DAG: Capability AtomicInt16CompareExchangeINTEL
; CHECK-DAG: Extension "SPV_INTEL_16bit_atomics"
; CHECK-DAG: %[[#TyInt16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#TyInt16Ptr:]] = OpTypePointer {{[a-zA-Z]+}} %[[#TyInt16]]
; CHECK-DAG: %[[#TyInt32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Value1:]] = OpConstant %[[#TyInt16]] 1{{$}}
; CHECK-DAG: %[[#Value42:]] = OpConstant %[[#TyInt16]] 42{{$}}
; CHECK-DAG: %[[#Const0:]] = OpConstantNull %[[#TyInt16]]
; CHECK-DAG: %[[#Int16Ptr:]] = OpVariable %[[#TyInt16Ptr]] CrossWorkgroup %[[#Const0]]
; CHECK-DAG: %[[#Scope:]] = OpConstantNull %[[#TyInt32]]
; CHECK-DAG: %[[#MemSeqCst:]] = OpConstant %[[#TyInt32]] 16{{$}}

; CHECK: OpAtomicExchange %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value42]]
; CHECK: OpAtomicCompareExchange %[[#TyInt16]] %[[#]] %[[#Scope]] %[[#MemSeqCst]] %[[#MemSeqCst]] %[[#Value42]] %[[#Value1]]


@val = common dso_local local_unnamed_addr addrspace(1) global i16 0, align 2

define dso_local spir_func void @test_atomic_int16_basic() local_unnamed_addr {
entry:
  %xchg = atomicrmw xchg ptr addrspace(1) @val, i16 42 seq_cst
  %cmpxchg = cmpxchg ptr addrspace(1) @val, i16 1, i16 42 seq_cst seq_cst
  ret void
}
