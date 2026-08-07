; Translating atomicrmw uinc_wrap/udec_wrap into a call to an imported helper is
; an AMD extension: there is no SPIR-V opcode for these, so a consumer has to
; recognize the helper by name to make sense of the module. Verify that a
; non-AMD target does not emit it, and instead falls back to the generic CmpXChg
; expansion. The AMD behaviour is covered by atomicrmw-uinc-udec-wrap.ll.
;
; --implicit-check-not applies over the whole module, unlike a CHECK-NOT, which
; would only cover the input up to the first positive match below.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=__translate_spirv_atomic
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=__translate_spirv_atomic
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@ui = common dso_local addrspace(1) global i32 0, align 4

; Both operations expand to an OpAtomicCompareExchange retry loop.
; CHECK: OpAtomicCompareExchange
define dso_local spir_func void @atomicrmw_uinc_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}

; CHECK: OpAtomicCompareExchange
define dso_local spir_func void @atomicrmw_udec_wrap() local_unnamed_addr {
entry:
  %0 = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret void
}
