; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_16bit_atomics %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: 16-bit integer atomic operations require the following SPIR-V extension: SPV_INTEL_16bit_atomics

; CHECK-DAG: Capability Int16
; CHECK-DAG: Capability Int16AtomicsINTEL
; CHECK-DAG: Extension "SPV_INTEL_16bit_atomics"
; CHECK-DAG: %[[#TyInt16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#TyInt16Ptr:]] = OpTypePointer {{[a-zA-Z]+}} %[[#TyInt16]]
; CHECK-DAG: %[[#TyInt32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Value5:]] = OpConstant %[[#TyInt16]] 5{{$}}
; CHECK-DAG: %[[#Const0:]] = OpConstantNull %[[#TyInt16]]
; CHECK-DAG: %[[#Int16Ptr:]] = OpVariable %[[#TyInt16Ptr]] CrossWorkgroup %[[#Const0]]
; CHECK-DAG: %[[#Scope:]] = OpConstantNull %[[#TyInt32]]
; CHECK-DAG: %[[#MemSeqCst:]] = OpConstant %[[#TyInt32]] 16{{$}}

; CHECK: OpAtomicIAdd %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]
; CHECK: OpAtomicISub %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]
; CHECK: OpAtomicUMin %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]
; CHECK: OpAtomicUMax %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]
; CHECK: OpAtomicSMin %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]
; CHECK: OpAtomicSMax %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]
; CHECK: OpAtomicAnd %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]
; CHECK: OpAtomicOr %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]
; CHECK: OpAtomicXor %[[#TyInt16]] %[[#Int16Ptr]] %[[#Scope]] %[[#MemSeqCst]] %[[#Value5]]


@val = common dso_local local_unnamed_addr addrspace(1) global i16 0, align 2

define dso_local spir_func void @test_atomic_int16_arithmetic() local_unnamed_addr {
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
