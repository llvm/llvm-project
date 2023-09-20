; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK:     %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Scope_Device:]] = OpConstant %[[#Int]] 1{{$}}
; CHECK-DAG: %[[#MemSem_Relaxed:]] = OpConstant %[[#Int]] 0
; CHECK-DAG: %[[#MemSem_Acquire:]] = OpConstant %[[#Int]] 2
; CHECK-DAG: %[[#MemSem_Release:]] = OpConstant %[[#Int]] 4{{$}}
; CHECK-DAG: %[[#MemSem_AcquireRelease:]] = OpConstant %[[#Int]] 8
; CHECK-DAG: %[[#MemSem_SequentiallyConsistent:]] = OpConstant %[[#Int]] 16
; CHECK-DAG: %[[#Value:]] = OpConstant %[[#Int]] 42
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#PointerType:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#FPPointerType:]] = OpTypePointer CrossWorkgroup %[[#Float]]
; CHECK-DAG: %[[#Pointer:]] = OpVariable %[[#PointerType]] CrossWorkgroup
; CHECK-DAG: %[[#FPPointer:]] = OpVariable %[[#FPPointerType]] CrossWorkgroup
; CHECK-DAG: %[[#FPValue:]] = OpConstant %[[#Float]] 1109917696

@ui = common dso_local addrspace(1) global i32 0, align 4
@f = common dso_local local_unnamed_addr addrspace(1) global float 0.000000e+00, align 4

define dso_local spir_func void @test_atomicrmw() local_unnamed_addr {
entry:
  %0 = atomicrmw xchg i32 addrspace(1)* @ui, i32 42 acq_rel
; CHECK: %[[#]] = OpAtomicExchange %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_AcquireRelease]] %[[#Value]]

  %1 = atomicrmw xchg float addrspace(1)* @f, float 42.000000e+00 seq_cst
; CHECK: %[[#]] = OpAtomicExchange %[[#Float]] %[[#FPPointer]] %[[#Scope_Device]] %[[#MemSem_SequentiallyConsistent]] %[[#FPValue]]

  %2 = atomicrmw add i32 addrspace(1)* @ui, i32 42 monotonic
; CHECK: %[[#]] = OpAtomicIAdd %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_Relaxed]] %[[#Value]]

  %3 = atomicrmw sub i32 addrspace(1)* @ui, i32 42 acquire
; CHECK: %[[#]] = OpAtomicISub %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_Acquire]] %[[#Value]]

  %4 = atomicrmw or i32 addrspace(1)* @ui, i32 42 release
; CHECK: %[[#]] = OpAtomicOr %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_Release]] %[[#Value]]

  %5 = atomicrmw xor i32 addrspace(1)* @ui, i32 42 acq_rel
; CHECK: %[[#]] = OpAtomicXor %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_AcquireRelease]] %[[#Value]]

  %6 = atomicrmw and i32 addrspace(1)* @ui, i32 42 seq_cst
; CHECK: %[[#]] = OpAtomicAnd %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_SequentiallyConsistent]] %[[#Value]]

  %7 = atomicrmw max i32 addrspace(1)* @ui, i32 42 monotonic
; CHECK: %[[#]] = OpAtomicSMax %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_Relaxed]] %[[#Value]]

  %8 = atomicrmw min i32 addrspace(1)* @ui, i32 42 acquire
; CHECK: %[[#]] = OpAtomicSMin %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_Acquire]] %[[#Value]]

  %9 = atomicrmw umax i32 addrspace(1)* @ui, i32 42 release
; CHECK: %[[#]] = OpAtomicUMax %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_Release]] %[[#Value]]

  %10 = atomicrmw umin i32 addrspace(1)* @ui, i32 42 acq_rel
; CHECK: %[[#]] = OpAtomicUMin %[[#Int]] %[[#Pointer]] %[[#Scope_Device]] %[[#MemSem_AcquireRelease]] %[[#Value]]

  ret void
}
