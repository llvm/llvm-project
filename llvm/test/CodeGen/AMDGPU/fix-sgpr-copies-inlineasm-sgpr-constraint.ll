; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck %s

; Verify behavior of inline asm operands with "s" (SGPR) constraints whose
; values arrive in VGPRs via calling convention.

; FIXME: The V2S copy heuristic in SIFixSGPRCopies converts these copies to
; VALU because INLINEASM is not classified as SALU, so SChain is empty and the
; score is zero. This leaves VGPRs where SGPRs are required, producing invalid
; assembly (buffer_store with VGPR resource descriptor, s_mov_b64 exec from
; VGPR). The correct behavior is to insert v_readfirstlane_b32 for each 32-bit
; sub-register.

; CHECK-LABEL: inlineasm_buffer_store_sgpr:
; CHECK-NOT: v_readfirstlane_b32
; CHECK: buffer_store_dwordx2 v[0:1], v2, v[4:7], 0 offen offset:0
; CHECK: s_mov_b64 exec, v[{{[0-9]+:[0-9]+}}]
define void @inlineasm_buffer_store_sgpr(<4 x half> %data, i32 %voffset, <4 x i32> %rsrc, i64 %save_exec) #0 {
entry:
  call void asm sideeffect
    "v_cmpx_le_u32 exec, 1, $4\0Abuffer_store_dwordx2 $0, $1, $2, 0 offen offset:$3\0As_mov_b64 exec, $5",
    "v,v,s,n,v,s,~{memory}"(
      <4 x half> %data, i32 %voffset, <4 x i32> %rsrc,
      i32 0, i32 1, i64 %save_exec)
  ret void
}

; Simpler test: just the 64-bit SGPR constraint with VGPR source.

; CHECK-LABEL: inlineasm_sgpr64_restore_exec:
; CHECK-NOT: v_readfirstlane_b32
; CHECK: s_mov_b64 exec, v[{{[0-9]+:[0-9]+}}]
define void @inlineasm_sgpr64_restore_exec(i64 %val) #0 {
entry:
  call void asm sideeffect "s_mov_b64 exec, $0", "s"(i64 %val)
  ret void
}

attributes #0 = { nounwind "target-cpu"="gfx950" }
