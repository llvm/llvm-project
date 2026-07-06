; Verify that atomicrmw uinc_wrap/udec_wrap correctly encode the
; storage-class semantics bit for different address spaces.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that atomicrmw uinc_wrap/udec_wrap correctly encode scopes. 
; CrossWorkgroupMemory = 0x200 = 512
; WorkgroupMemory      = 0x100 = 256
; AcquireRelease       = 0x008 =   8
;   -> with CrossWorkgroup: 520
;   -> with Workgroup:      264

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#MemSem_AcqRel_CW:]] = OpConstant %[[#Int]] 520
; CHECK-DAG: %[[#MemSem_AcqRel_WG:]] = OpConstant %[[#Int]] 264

@g_cw = common dso_local addrspace(1) global i32 0, align 4
@g_wg = common dso_local addrspace(3) global i32 0, align 4

define dso_local spir_func void @uinc_wrap_crossworkgroup() {
entry:
  ; CHECK: OpAtomicIIncrement %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_AcqRel_CW]]
  %0 = atomicrmw uinc_wrap ptr addrspace(1) @g_cw, i32 1 acq_rel
  ret void
}

define dso_local spir_func void @uinc_wrap_workgroup() {
entry:
  ; CHECK: OpAtomicIIncrement %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_AcqRel_WG]]
  %0 = atomicrmw uinc_wrap ptr addrspace(3) @g_wg, i32 1 acq_rel
  ret void
}

define dso_local spir_func void @udec_wrap_crossworkgroup() {
entry:
  ; CHECK: OpAtomicIDecrement %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_AcqRel_CW]]
  %0 = atomicrmw udec_wrap ptr addrspace(1) @g_cw, i32 1 acq_rel
  ret void
}

define dso_local spir_func void @udec_wrap_workgroup() {
entry:
  ; CHECK: OpAtomicIDecrement %[[#Int]] %{{[0-9]+}} %{{[0-9]+}} %[[#MemSem_AcqRel_WG]]
  %0 = atomicrmw udec_wrap ptr addrspace(3) @g_wg, i32 1 acq_rel
  ret void
}
