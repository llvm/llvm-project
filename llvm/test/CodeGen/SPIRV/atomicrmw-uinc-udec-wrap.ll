; Verify that atomicrmw uinc_wrap/udec_wrap lower to OpFunctionCall to
; __spirv_AtomicUIncWrap/__spirv_AtomicUDecWrap with Import linkage,
; rather than being expanded to a CmpXChg loop.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#MemSem_SeqCst:]] = OpConstant %[[#Int]] 528
; CHECK-DAG: %[[#WrapVal:]] = OpConstant %[[#Int]] 42
; CHECK-DAG: %[[#Scope_CrossDevice:]] = OpConstantNull %[[#Int]]

; CHECK-DAG: OpDecorate %[[#UIncWrapFn:]] LinkageAttributes "__spirv_AtomicUIncWrap" Import
; CHECK-DAG: OpDecorate %[[#UDecWrapFn:]] LinkageAttributes "__spirv_AtomicUDecWrap" Import

@ui = common dso_local addrspace(1) global i32 0, align 4

; CHECK: OpFunctionCall %[[#Int]] %[[#UIncWrapFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_SeqCst]] %[[#WrapVal]]
define dso_local spir_func void @atomicrmw_uinc_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}

; CHECK: OpFunctionCall %[[#Int]] %[[#UDecWrapFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_SeqCst]] %[[#WrapVal]]
define dso_local spir_func void @atomicrmw_udec_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}
