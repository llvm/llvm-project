; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -stop-after=si-insert-skips < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: name: syncscopes
; GCN: FLAT_STORE_DWORD killed renamable $vgpr1_vgpr2, killed renamable $vgpr0, 0, 0, implicit $exec, implicit $flat_scr :: (store syncscope("agent") seq_cst 4 into %ir.agent_out)
; GCN: FLAT_STORE_DWORD killed renamable $vgpr4_vgpr5, killed renamable $vgpr3, 0, 0, implicit $exec, implicit $flat_scr :: (store syncscope("workgroup") seq_cst 4 into %ir.workgroup_out)
; GCN: FLAT_STORE_DWORD killed renamable $vgpr7_vgpr8, killed renamable $vgpr6, 0, 0, implicit $exec, implicit $flat_scr :: (store syncscope("wavefront") seq_cst 4 into %ir.wavefront_out)
define void @syncscopes(
    i32 %agent,
    i32* %agent_out,
    i32 %workgroup,
    i32* %workgroup_out,
    i32 %wavefront,
    i32* %wavefront_out) {
entry:
  store atomic i32 %agent, i32* %agent_out syncscope("agent") seq_cst, align 4
  store atomic i32 %workgroup, i32* %workgroup_out syncscope("workgroup") seq_cst, align 4
  store atomic i32 %wavefront, i32* %wavefront_out syncscope("wavefront") seq_cst, align 4
  ret void
}
