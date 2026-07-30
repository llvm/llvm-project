; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that atomicrmw uinc_wrap/udec_wrap correctly encode scopes.

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Scope_CrossDevice:]] = OpConstantNull %[[#Int]]
; CHECK-DAG: %[[#Scope_Device:]] = OpConstant %[[#Int]] 1{{$}}
; CHECK-DAG: %[[#Scope_Workgroup:]] = OpConstant %[[#Int]] 2{{$}}
; CHECK-DAG: %[[#Scope_Subgroup:]] = OpConstant %[[#Int]] 3{{$}}
; CHECK-DAG: %[[#Scope_Invocation:]] = OpConstant %[[#Int]] 4{{$}}
; CHECK-DAG: %[[#MemSem_SeqCst:]] = OpConstant %[[#Int]] 528{{$}}

; CHECK-DAG: OpDecorate %[[#UIncFn:]] LinkageAttributes "__spirv_AtomicUIncWrap" Import
; CHECK-DAG: OpDecorate %[[#UDecFn:]] LinkageAttributes "__spirv_AtomicUDecWrap" Import

@ui = common dso_local addrspace(1) global i32 0, align 4

define dso_local spir_func void @uinc_wrap_scopes() {
entry:
  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_SeqCst]]
  %0 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope_Device]] %[[#MemSem_SeqCst]]
  %1 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 syncscope("device") seq_cst

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope_Workgroup]] %[[#MemSem_SeqCst]]
  %2 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 syncscope("workgroup") seq_cst

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope_Subgroup]] %[[#MemSem_SeqCst]]
  %3 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 syncscope("subgroup") seq_cst

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope_Invocation]] %[[#MemSem_SeqCst]]
  %4 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 syncscope("singlethread") seq_cst

  ret void
}

define dso_local spir_func void @udec_wrap_scopes() {
entry:
  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope_CrossDevice]] %[[#MemSem_SeqCst]]
  %0 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope_Device]] %[[#MemSem_SeqCst]]
  %1 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("device") seq_cst

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope_Workgroup]] %[[#MemSem_SeqCst]]
  %2 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("workgroup") seq_cst

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope_Subgroup]] %[[#MemSem_SeqCst]]
  %3 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("subgroup") seq_cst

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope_Invocation]] %[[#MemSem_SeqCst]]
  %4 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("singlethread") seq_cst

  ret void
}
