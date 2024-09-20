; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK:     %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Scope_CrossDevice:]] = OpConstant %[[#Int]] 0
; CHECK-DAG: %[[#Value:]] = OpConstant %[[#Int]] 42
; CHECK-DAG: %[[#FPValue:]] = OpConstant %[[#Float]] 42
; CHECK-DAG: %[[#Scope_Invocation:]] = OpConstant %[[#Int]] 4
; CHECK-DAG: %[[#MemSem_SeqCst:]] = OpConstant %[[#Int]] 16
; CHECK-DAG: %[[#Scope_Subgroup:]] = OpConstant %[[#Int]] 3
; CHECK-DAG: %[[#Scope_Workgroup:]] = OpConstant %[[#Int]] 2
; CHECK-DAG: %[[#Scope_Device:]] = OpConstant %[[#Int]] 1
; CHECK-DAG: %[[#PointerType:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#FPPointerType:]] = OpTypePointer CrossWorkgroup %[[#Float]]
; CHECK-DAG: %[[#Pointer:]] = OpVariable %[[#PointerType]] CrossWorkgroup
; CHECK-DAG: %[[#FPPointer:]] = OpVariable %[[#FPPointerType]] CrossWorkgroup

@ui = common dso_local addrspace(1) global i32 0, align 4
@f = common dso_local local_unnamed_addr addrspace(1) global float 0.000000e+00, align 4

define dso_local spir_func void @test_singlethread_atomicrmw() local_unnamed_addr {
entry:
  %0 = atomicrmw xchg i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %1 = atomicrmw xchg float addrspace(1)* @f, float 42.000000e+00 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Float:]] %[[#FPPointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#FPValue:]]
  %2 = atomicrmw add i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicIAdd %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %3 = atomicrmw sub i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicISub %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %4 = atomicrmw or i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicOr %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %5 = atomicrmw xor i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicXor %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %6 = atomicrmw and i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicAnd %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %7 = atomicrmw max i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicSMax %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %8 = atomicrmw min i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicSMin %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %9 = atomicrmw umax i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicUMax %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %10 = atomicrmw umin i32 addrspace(1)* @ui, i32 42 syncscope("singlethread") seq_cst
  ; CHECK: %[[#]] = OpAtomicUMin %[[#Int]] %[[#Pointer:]] %[[#Scope_Invocation:]] %[[#MemSem_SeqCst:]] %[[#Value:]]

  ret void
}

define dso_local spir_func void @test_subgroup_atomicrmw() local_unnamed_addr {
entry:
  %0 = atomicrmw xchg i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %1 = atomicrmw xchg float addrspace(1)* @f, float 42.000000e+00 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Float:]] %[[#FPPointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#FPValue:]]
  %2 = atomicrmw add i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicIAdd %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %3 = atomicrmw sub i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicISub %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %4 = atomicrmw or i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicOr %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %5 = atomicrmw xor i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicXor %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %6 = atomicrmw and i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicAnd %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %7 = atomicrmw max i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicSMax %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %8 = atomicrmw min i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicSMin %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %9 = atomicrmw umax i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicUMax %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %10 = atomicrmw umin i32 addrspace(1)* @ui, i32 42 syncscope("subgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicUMin %[[#Int]] %[[#Pointer:]] %[[#Scope_Subgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]

  ret void
}

define dso_local spir_func void @test_workgroup_atomicrmw() local_unnamed_addr {
entry:
  %0 = atomicrmw xchg i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %1 = atomicrmw xchg float addrspace(1)* @f, float 42.000000e+00 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Float:]] %[[#FPPointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#FPValue:]]
  %2 = atomicrmw add i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicIAdd %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %3 = atomicrmw sub i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicISub %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %4 = atomicrmw or i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicOr %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %5 = atomicrmw xor i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicXor %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %6 = atomicrmw and i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicAnd %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %7 = atomicrmw max i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicSMax %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %8 = atomicrmw min i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicSMin %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %9 = atomicrmw umax i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicUMax %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %10 = atomicrmw umin i32 addrspace(1)* @ui, i32 42 syncscope("workgroup") seq_cst
  ; CHECK: %[[#]] = OpAtomicUMin %[[#Int]] %[[#Pointer:]] %[[#Scope_Workgroup:]] %[[#MemSem_SeqCst:]] %[[#Value:]]

  ret void
}

define dso_local spir_func void @test_device_atomicrmw() local_unnamed_addr {
entry:
  %0 = atomicrmw xchg i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %1 = atomicrmw xchg float addrspace(1)* @f, float 42.000000e+00 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Float:]] %[[#FPPointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#FPValue:]]
  %2 = atomicrmw add i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicIAdd %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %3 = atomicrmw sub i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicISub %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %4 = atomicrmw or i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicOr %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %5 = atomicrmw xor i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicXor %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %6 = atomicrmw and i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicAnd %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %7 = atomicrmw max i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicSMax %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %8 = atomicrmw min i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicSMin %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %9 = atomicrmw umax i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicUMax %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %10 = atomicrmw umin i32 addrspace(1)* @ui, i32 42 syncscope("device") seq_cst
  ; CHECK: %[[#]] = OpAtomicUMin %[[#Int]] %[[#Pointer:]] %[[#Scope_Device:]] %[[#MemSem_SeqCst:]] %[[#Value:]]

  ret void
}

define dso_local spir_func void @test_all_svm_devices_atomicrmw() local_unnamed_addr {
entry:
  %0 = atomicrmw xchg i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %1 = atomicrmw xchg float addrspace(1)* @f, float 42.000000e+00 seq_cst
  ; CHECK: %[[#]] = OpAtomicExchange %[[#Float:]] %[[#FPPointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#FPValue:]]
  %2 = atomicrmw add i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicIAdd %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %3 = atomicrmw sub i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicISub %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %4 = atomicrmw or i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicOr %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %5 = atomicrmw xor i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicXor %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %6 = atomicrmw and i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicAnd %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %7 = atomicrmw max i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicSMax %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %8 = atomicrmw min i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicSMin %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %9 = atomicrmw umax i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicUMax %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]
  %10 = atomicrmw umin i32 addrspace(1)* @ui, i32 42 seq_cst
  ; CHECK: %[[#]] = OpAtomicUMin %[[#Int]] %[[#Pointer:]] %[[#Scope_CrossDevice:]] %[[#MemSem_SeqCst:]] %[[#Value:]]

  ret void
}
