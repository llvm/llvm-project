; Verify that atomicrmw uinc_wrap/udec_wrap lower to OpAtomicIIncrement/
; OpAtomicIDecrement with a MaxByteOffsetId decoration carrying the wrap
; operand, rather than being expanded to a CmpXChg loop.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PointerType:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#MemSem_SeqCst:]] = OpConstant %[[#Int]] 528
; CHECK-DAG: %[[#Scope_CrossDevice:]] = OpConstantNull %[[#Int]]
; CHECK-DAG: %[[#Pointer:]] = OpVariable %[[#PointerType]] CrossWorkgroup

; CHECK-DAG: OpDecorate %[[#Inc:]] MaxByteOffsetId %[[#WrapVal:]]
; CHECK-DAG: OpDecorate %[[#Dec:]] MaxByteOffsetId %[[#WrapVal2:]]

@ui = common dso_local addrspace(1) global i32 0, align 4

; CHECK: %[[#Inc]] = OpAtomicIIncrement %[[#Int]] %[[#Pointer]] %[[#Scope_CrossDevice]] %[[#MemSem_SeqCst]]
define dso_local spir_func void @atomicrmw_uinc_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}

; CHECK: %[[#Dec]] = OpAtomicIDecrement %[[#Int]] %[[#Pointer]] %[[#Scope_CrossDevice]] %[[#MemSem_SeqCst]]
define dso_local spir_func void @atomicrmw_udec_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}
