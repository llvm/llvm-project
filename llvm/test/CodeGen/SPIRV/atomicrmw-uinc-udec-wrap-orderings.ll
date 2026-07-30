; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that atomicrmw uinc_wrap/udec_wrap correctly encode memory
; orderings in the function call arguments.
; CrossWorkgroupMemory = 0x200 = 512
; Monotonic (Relaxed) = 0x000 -> with CrossWorkgroup: 512
; Acquire              = 0x002 -> with CrossWorkgroup: 514
; Release              = 0x004 -> with CrossWorkgroup: 516
; AcquireRelease       = 0x008 -> with CrossWorkgroup: 520
; SequentiallyConsistent = 0x010 -> with CrossWorkgroup: 528

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Scope:]] = OpConstantNull %[[#Int]]
; CHECK-DAG: %[[#MemSem_Monotonic:]] = OpConstant %[[#Int]] 512
; CHECK-DAG: %[[#MemSem_Acquire:]] = OpConstant %[[#Int]] 514
; CHECK-DAG: %[[#MemSem_Release:]] = OpConstant %[[#Int]] 516
; CHECK-DAG: %[[#MemSem_AcqRel:]] = OpConstant %[[#Int]] 520
; CHECK-DAG: %[[#MemSem_SeqCst:]] = OpConstant %[[#Int]] 528

; CHECK-DAG: OpDecorate %[[#UIncFn:]] LinkageAttributes "__spirv_AtomicUIncWrap" Import
; CHECK-DAG: OpDecorate %[[#UDecFn:]] LinkageAttributes "__spirv_AtomicUDecWrap" Import

@ui = common dso_local addrspace(1) global i32 0, align 4

define dso_local spir_func void @uinc_wrap_orderings() {
entry:
  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope]] %[[#MemSem_Monotonic]]
  %0 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope]] %[[#MemSem_Acquire]]
  %1 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 acquire

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope]] %[[#MemSem_Release]]
  %2 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 release

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope]] %[[#MemSem_AcqRel]]
  %3 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 acq_rel

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UIncFn]] %[[#]] %[[#Scope]] %[[#MemSem_SeqCst]]
  %4 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst

  ret void
}

define dso_local spir_func void @udec_wrap_orderings() {
entry:
  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope]] %[[#MemSem_Monotonic]]
  %0 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 monotonic

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope]] %[[#MemSem_Acquire]]
  %1 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 acquire

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope]] %[[#MemSem_Release]]
  %2 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 release

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope]] %[[#MemSem_AcqRel]]
  %3 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 acq_rel

  ; CHECK: OpFunctionCall %[[#Int]] %[[#UDecFn]] %[[#]] %[[#Scope]] %[[#MemSem_SeqCst]]
  %4 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst

  ret void
}
