; RUN: llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX908 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX900 %s

define void @foo() {
bb:
  ret void
}

; FIXME: We spill v40 into AGPR, but still save and restore FP
; which is not needed in this case.

; GCN-LABEL: {{^}}caller:

; GCN:     s_mov_b32 [[TMP_SGPR:s[0-9]+]], s33
; GCN:     s_mov_b32 s33, s32
; GFX900:     buffer_store_dword
; GFX908-DAG: v_accvgpr_write_b32
; GCN:        s_swappc_b64
; GFX900:     buffer_load_dword
; GFX908:     v_accvgpr_read_b32
; GCN:        s_mov_b32 s33, [[TMP_SGPR]]
define i64 @caller() {
bb:
  call void asm sideeffect "", "~{v40}" ()
  tail call void @foo()
  ret i64 0
}
