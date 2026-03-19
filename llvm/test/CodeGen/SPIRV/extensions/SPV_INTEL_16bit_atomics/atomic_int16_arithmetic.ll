; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: 16-bit integer atomic operations require the following SPIR-V extension: SPV_INTEL_16bit_atomics

; CHECK-DAG: Capability Int16
; CHECK-DAG: Capability Int16AtomicsINTEL
; CHECK-DAG: Extension "SPV_INTEL_16bit_atomics"
; CHECK-DAG: %[[#TyInt16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#TyInt32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TyInt16Ptr:]] = OpTypePointer CrossWorkgroup %[[#TyInt16]]
; CHECK-DAG: %[[#Const5:]] = OpConstant %[[#TyInt16]] 5{{$}}
; CHECK-DAG: %[[#Const0:]] = OpConstantNull %[[#TyInt16]]
; CHECK-DAG: %[[#Scope:]] = OpConstantNull %[[#TyInt32]]
; CHECK-DAG: %[[#MemSeqCst:]] = OpConstant %[[#TyInt32]] 16{{$}}
; CHECK-DAG: %[[#Val:]] = OpVariable %[[#TyInt16Ptr]] CrossWorkgroup %[[#Const0]]

; CHECK: OpAtomicIAdd %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]
; CHECK: OpAtomicISub %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]
; CHECK: OpAtomicUMin %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]
; CHECK: OpAtomicUMax %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]
; CHECK: OpAtomicSMin %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]
; CHECK: OpAtomicSMax %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]
; CHECK: OpAtomicAnd %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]
; CHECK: OpAtomicOr %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]
; CHECK: OpAtomicXor %[[#TyInt16]] %[[#Val]] %[[#Scope]] %[[#MemSeqCst]] %[[#Const5]]


@val = private addrspace(1) global i16 0

define spir_func void @test_atomic_int16_arithmetic() {
entry:
  %add = atomicrmw add ptr addrspace(1) @val, i16 5 seq_cst
  %sub = atomicrmw sub ptr addrspace(1) @val, i16 5 seq_cst
  %umin = atomicrmw umin ptr addrspace(1) @val, i16 5 seq_cst
  %umax = atomicrmw umax ptr addrspace(1) @val, i16 5 seq_cst
  %smin = atomicrmw min ptr addrspace(1) @val, i16 5 seq_cst
  %smax = atomicrmw max ptr addrspace(1) @val, i16 5 seq_cst
  %and = atomicrmw and ptr addrspace(1) @val, i16 5 seq_cst
  %or = atomicrmw or ptr addrspace(1) @val, i16 5 seq_cst
  %xor = atomicrmw xor ptr addrspace(1) @val, i16 5 seq_cst
  ret void
}
