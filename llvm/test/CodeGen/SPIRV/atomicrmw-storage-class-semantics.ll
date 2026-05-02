; Verify that selectAtomicRMW emits MemorySemantics with the storage-class
; bit OR'd into the ordering bits, as required by the SPIR-V spec
; (section 3.32, Memory Semantics): atomic ops must combine the ordering
; bits (Acquire/Release/AcquireRelease/SequentiallyConsistent) with the
; relevant storage-class bit (UniformMemory, WorkgroupMemory,
; CrossWorkgroupMemory, ...).

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CrossWorkgroupMemory = 0x200 = 512
; WorkgroupMemory      = 0x100 = 256
; AcquireRelease       = 0x008 =   8 -> with CrossWorkgroup: 520, with Workgroup: 264
; SequentiallyConsistent = 0x010 = 16 -> with CrossWorkgroup: 528, with Workgroup: 272
; Acquire              = 0x002 =   2 -> with CrossWorkgroup: 514

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#MemSem_AcqRel_CW:]] = OpConstant %[[#Int]] 520
; CHECK-DAG: %[[#MemSem_SeqCst_CW:]] = OpConstant %[[#Int]] 528
; CHECK-DAG: %[[#MemSem_Acquire_CW:]] = OpConstant %[[#Int]] 514
; CHECK-DAG: %[[#MemSem_AcqRel_WG:]] = OpConstant %[[#Int]] 264
; CHECK-DAG: %[[#MemSem_SeqCst_WG:]] = OpConstant %[[#Int]] 272

@g_cw = common dso_local addrspace(1) global i32 0, align 4
@g_wg = common dso_local addrspace(3) global i32 0, align 4

define dso_local spir_func void @test_crossworkgroup() {
entry:
  ; CrossWorkgroup pointer + acq_rel -> MemSem 520 (8 | 0x200)
  ; CHECK: OpAtomicIAdd %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_AcqRel_CW]] %{{[0-9]+}}
  %0 = atomicrmw add ptr addrspace(1) @g_cw, i32 1 acq_rel

  ; CrossWorkgroup pointer + seq_cst -> MemSem 528 (16 | 0x200)
  ; CHECK: OpAtomicExchange %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_SeqCst_CW]] %{{[0-9]+}}
  %1 = atomicrmw xchg ptr addrspace(1) @g_cw, i32 7 seq_cst

  ; CrossWorkgroup pointer + acquire -> MemSem 514 (2 | 0x200)
  ; CHECK: OpAtomicOr %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_Acquire_CW]] %{{[0-9]+}}
  %2 = atomicrmw or ptr addrspace(1) @g_cw, i32 3 acquire

  ret void
}

define dso_local spir_func void @test_workgroup() {
entry:
  ; Workgroup pointer + acq_rel -> MemSem 264 (8 | 0x100)
  ; CHECK: OpAtomicIAdd %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_AcqRel_WG]] %{{[0-9]+}}
  %0 = atomicrmw add ptr addrspace(3) @g_wg, i32 1 acq_rel

  ; Workgroup pointer + seq_cst -> MemSem 272 (16 | 0x100)
  ; CHECK: OpAtomicExchange %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_SeqCst_WG]] %{{[0-9]+}}
  %1 = atomicrmw xchg ptr addrspace(3) @g_wg, i32 7 seq_cst

  ret void
}
