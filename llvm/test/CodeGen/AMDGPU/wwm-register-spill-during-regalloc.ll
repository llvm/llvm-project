; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -stop-after=virtregrewriter,1 --verify-machineinstrs -o - %s | FileCheck -check-prefix=WWM-SPILL %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -O0 -stop-after=regallocfast,1 --verify-machineinstrs -o - %s | FileCheck -check-prefix=WWM-SPILL-O0 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 --verify-machineinstrs -o - %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -O0 --verify-machineinstrs -o - %s | FileCheck -check-prefix=GCN-O0 %s

; Test whole-wave register spilling.

; In the testcase, the return address registers (SGPR30_SGPR31) should be preserved across the call.
; Since the test limits the VGPR numbers, they are all in the call-clobber (scratch) range and RA should
; spill any VGPR borrowed for spilling SGPRs. The writelane/readlane instructions that spill/restore
; SGPRs into/from VGPR are whole-wave operations and hence the VGPRs involved in such operations require
; whole-wave spilling.

define void @test() #0 {
; WWM-SPILL-LABEL: name: test
; WWM-SPILL: bb.0 (%ir-block.0):
; WWM-SPILL-NEXT:   liveins: $sgpr12, $sgpr13, $sgpr14, $sgpr15, $sgpr30, $sgpr31, $vgpr31, $sgpr4_sgpr5, $sgpr6_sgpr7, $sgpr8_sgpr9, $sgpr10_sgpr11
; WWM-SPILL-NEXT: {{  $}}
; WWM-SPILL-NEXT:   renamable $vgpr0 = IMPLICIT_DEF
; WWM-SPILL-NEXT:   renamable $vgpr0 = V_WRITELANE_B32 killed $sgpr30, 0, killed $vgpr0
; WWM-SPILL-NEXT:   renamable $vgpr0 = V_WRITELANE_B32 killed $sgpr31, 1, killed $vgpr0
; WWM-SPILL-NEXT:   SI_SPILL_WWM_V32_SAVE killed $vgpr0, %stack.2, $sgpr32, 0, implicit $exec :: (store (s32) into %stack.2, addrspace 5)
; WWM-SPILL-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def dead $scc, implicit-def $sgpr32, implicit $sgpr32
; WWM-SPILL-NEXT:   renamable $sgpr16_sgpr17 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @ext_func + 4, target-flags(amdgpu-gotprel32-hi) @ext_func + 12, implicit-def dead $scc
; WWM-SPILL-NEXT:   renamable $sgpr16_sgpr17 = S_LOAD_DWORDX2_IMM killed renamable $sgpr16_sgpr17, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; WWM-SPILL-NEXT:   dead $sgpr30_sgpr31 = SI_CALL killed renamable $sgpr16_sgpr17, @ext_func, csr_amdgpu, implicit $sgpr4_sgpr5, implicit $sgpr6_sgpr7, implicit $sgpr8_sgpr9, implicit $sgpr10_sgpr11, implicit $sgpr12, implicit $sgpr13, implicit $sgpr14, implicit $sgpr15, implicit $vgpr31, implicit $sgpr0_sgpr1_sgpr2_sgpr3
; WWM-SPILL-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def dead $scc, implicit-def $sgpr32, implicit $sgpr32
; WWM-SPILL-NEXT:   renamable $vgpr0 = SI_SPILL_WWM_V32_RESTORE %stack.2, $sgpr32, 0, implicit $exec :: (load (s32) from %stack.2, addrspace 5)
; WWM-SPILL-NEXT:   $sgpr31 = V_READLANE_B32 $vgpr0, 1
; WWM-SPILL-NEXT:   $sgpr30 = V_READLANE_B32 killed $vgpr0, 0
; WWM-SPILL-NEXT:   SI_RETURN
;
; WWM-SPILL-O0-LABEL: name: test
; WWM-SPILL-O0: bb.0 (%ir-block.0):
; WWM-SPILL-O0-NEXT:   liveins: $sgpr12, $sgpr13, $sgpr14, $sgpr15, $sgpr30, $sgpr31, $vgpr31, $sgpr4_sgpr5, $sgpr6_sgpr7, $sgpr8_sgpr9, $sgpr10_sgpr11
; WWM-SPILL-O0-NEXT: {{  $}}
; WWM-SPILL-O0-NEXT:   renamable $vgpr0 = IMPLICIT_DEF
; WWM-SPILL-O0-NEXT:   renamable $vgpr0 = V_WRITELANE_B32 killed $sgpr30, 0, $vgpr0
; WWM-SPILL-O0-NEXT:   renamable $vgpr0 = V_WRITELANE_B32 killed $sgpr31, 1, $vgpr0
; WWM-SPILL-O0-NEXT:   SI_SPILL_WWM_V32_SAVE $vgpr0, %stack.2, $sgpr32, 0, implicit $exec :: (store (s32) into %stack.2, addrspace 5)
; WWM-SPILL-O0-NEXT:   renamable $vgpr0 = COPY $vgpr31
; WWM-SPILL-O0-NEXT:   ADJCALLSTACKUP 0, 0, implicit-def dead $scc, implicit-def $sgpr32, implicit $sgpr32
; WWM-SPILL-O0-NEXT:   renamable $sgpr16_sgpr17 = SI_PC_ADD_REL_OFFSET target-flags(amdgpu-gotprel32-lo) @ext_func + 4, target-flags(amdgpu-gotprel32-hi) @ext_func + 12, implicit-def dead $scc
; WWM-SPILL-O0-NEXT:   renamable $sgpr16_sgpr17 = S_LOAD_DWORDX2_IMM killed renamable $sgpr16_sgpr17, 0, 0 :: (dereferenceable invariant load (s64) from got, addrspace 4)
; WWM-SPILL-O0-NEXT:   renamable $sgpr20_sgpr21_sgpr22_sgpr23 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
; WWM-SPILL-O0-NEXT:   $vgpr31 = COPY killed renamable $vgpr0
; WWM-SPILL-O0-NEXT:   $sgpr0_sgpr1_sgpr2_sgpr3 = COPY killed renamable $sgpr20_sgpr21_sgpr22_sgpr23
; WWM-SPILL-O0-NEXT:   dead $sgpr30_sgpr31 = SI_CALL killed renamable $sgpr16_sgpr17, @ext_func, csr_amdgpu, implicit killed $sgpr4_sgpr5, implicit killed $sgpr6_sgpr7, implicit killed $sgpr8_sgpr9, implicit killed $sgpr10_sgpr11, implicit killed $sgpr12, implicit killed $sgpr13, implicit killed $sgpr14, implicit killed $sgpr15, implicit $vgpr31, implicit $sgpr0_sgpr1_sgpr2_sgpr3
; WWM-SPILL-O0-NEXT:   $vgpr0 = SI_SPILL_WWM_V32_RESTORE %stack.2, $sgpr32, 0, implicit $exec :: (load (s32) from %stack.2, addrspace 5)
; WWM-SPILL-O0-NEXT:   ADJCALLSTACKDOWN 0, 0, implicit-def dead $scc, implicit-def $sgpr32, implicit $sgpr32
; WWM-SPILL-O0-NEXT:   dead $sgpr31 = V_READLANE_B32 $vgpr0, 1
; WWM-SPILL-O0-NEXT:   dead $sgpr30 = V_READLANE_B32 killed $vgpr0, 0
; WWM-SPILL-O0-NEXT:   SI_RETURN
;
; GCN-LABEL: test:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s16, s33
; GCN-NEXT:    s_mov_b32 s33, s32
; GCN-NEXT:    s_xor_saveexec_b64 s[18:19], -1
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GCN-NEXT:    s_mov_b64 exec, s[18:19]
; GCN-NEXT:    v_mov_b32_e32 v1, s34
; GCN-NEXT:    buffer_store_dword v1, off, s[0:3], s33 offset:8 ; 4-byte Folded Spill
; GCN-NEXT:    v_mov_b32_e32 v1, s35
; GCN-NEXT:    ; implicit-def: $vgpr0
; GCN-NEXT:    buffer_store_dword v1, off, s[0:3], s33 offset:12 ; 4-byte Folded Spill
; GCN-NEXT:    v_mov_b32_e32 v1, s16
; GCN-NEXT:    v_writelane_b32 v0, s30, 0
; GCN-NEXT:    buffer_store_dword v1, off, s[0:3], s33 offset:16 ; 4-byte Folded Spill
; GCN-NEXT:    s_addk_i32 s32, 0x800
; GCN-NEXT:    v_writelane_b32 v0, s31, 1
; GCN-NEXT:    s_or_saveexec_b64 s[34:35], -1
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s33 ; 4-byte Folded Spill
; GCN-NEXT:    s_mov_b64 exec, s[34:35]
; GCN-NEXT:    s_getpc_b64 s[16:17]
; GCN-NEXT:    s_add_u32 s16, s16, ext_func@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s17, s17, ext_func@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[16:17], s[16:17], 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[16:17]
; GCN-NEXT:    s_or_saveexec_b64 s[34:35], -1
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s33 ; 4-byte Folded Reload
; GCN-NEXT:    s_mov_b64 exec, s[34:35]
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_readlane_b32 s31, v0, 1
; GCN-NEXT:    v_readlane_b32 s30, v0, 0
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:8 ; 4-byte Folded Reload
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_readfirstlane_b32 s34, v0
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:12 ; 4-byte Folded Reload
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_readfirstlane_b32 s35, v0
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:16 ; 4-byte Folded Reload
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_readfirstlane_b32 s4, v0
; GCN-NEXT:    s_xor_saveexec_b64 s[6:7], -1
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
; GCN-NEXT:    s_mov_b64 exec, s[6:7]
; GCN-NEXT:    s_addk_i32 s32, 0xf800
; GCN-NEXT:    s_mov_b32 s33, s4
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
;
; GCN-O0-LABEL: test:
; GCN-O0:       ; %bb.0:
; GCN-O0-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-O0-NEXT:    s_mov_b32 s16, s33
; GCN-O0-NEXT:    s_mov_b32 s33, s32
; GCN-O0-NEXT:    s_xor_saveexec_b64 s[18:19], -1
; GCN-O0-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:4 ; 4-byte Folded Spill
; GCN-O0-NEXT:    s_mov_b64 exec, s[18:19]
; GCN-O0-NEXT:    v_mov_b32_e32 v1, s34
; GCN-O0-NEXT:    buffer_store_dword v1, off, s[0:3], s33 offset:8 ; 4-byte Folded Spill
; GCN-O0-NEXT:    v_mov_b32_e32 v1, s35
; GCN-O0-NEXT:    buffer_store_dword v1, off, s[0:3], s33 offset:12 ; 4-byte Folded Spill
; GCN-O0-NEXT:    v_mov_b32_e32 v1, s16
; GCN-O0-NEXT:    buffer_store_dword v1, off, s[0:3], s33 offset:16 ; 4-byte Folded Spill
; GCN-O0-NEXT:    s_add_i32 s32, s32, 0x800
; GCN-O0-NEXT:    ; implicit-def: $vgpr0
; GCN-O0-NEXT:    v_writelane_b32 v0, s30, 0
; GCN-O0-NEXT:    v_writelane_b32 v0, s31, 1
; GCN-O0-NEXT:    s_or_saveexec_b64 s[34:35], -1
; GCN-O0-NEXT:    buffer_store_dword v0, off, s[0:3], s33 ; 4-byte Folded Spill
; GCN-O0-NEXT:    s_mov_b64 exec, s[34:35]
; GCN-O0-NEXT:    v_mov_b32_e32 v0, v31
; GCN-O0-NEXT:    s_getpc_b64 s[16:17]
; GCN-O0-NEXT:    s_add_u32 s16, s16, ext_func@gotpcrel32@lo+4
; GCN-O0-NEXT:    s_addc_u32 s17, s17, ext_func@gotpcrel32@hi+12
; GCN-O0-NEXT:    s_load_dwordx2 s[16:17], s[16:17], 0x0
; GCN-O0-NEXT:    s_mov_b64 s[22:23], s[2:3]
; GCN-O0-NEXT:    s_mov_b64 s[20:21], s[0:1]
; GCN-O0-NEXT:    v_mov_b32_e32 v31, v0
; GCN-O0-NEXT:    s_mov_b64 s[0:1], s[20:21]
; GCN-O0-NEXT:    s_mov_b64 s[2:3], s[22:23]
; GCN-O0-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-O0-NEXT:    s_swappc_b64 s[30:31], s[16:17]
; GCN-O0-NEXT:    s_or_saveexec_b64 s[34:35], -1
; GCN-O0-NEXT:    buffer_load_dword v0, off, s[0:3], s33 ; 4-byte Folded Reload
; GCN-O0-NEXT:    s_mov_b64 exec, s[34:35]
; GCN-O0-NEXT:    s_waitcnt vmcnt(0)
; GCN-O0-NEXT:    v_readlane_b32 s31, v0, 1
; GCN-O0-NEXT:    v_readlane_b32 s30, v0, 0
; GCN-O0-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:8 ; 4-byte Folded Reload
; GCN-O0-NEXT:    s_waitcnt vmcnt(0)
; GCN-O0-NEXT:    v_readfirstlane_b32 s34, v0
; GCN-O0-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:12 ; 4-byte Folded Reload
; GCN-O0-NEXT:    s_waitcnt vmcnt(0)
; GCN-O0-NEXT:    v_readfirstlane_b32 s35, v0
; GCN-O0-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:16 ; 4-byte Folded Reload
; GCN-O0-NEXT:    s_waitcnt vmcnt(0)
; GCN-O0-NEXT:    v_readfirstlane_b32 s4, v0
; GCN-O0-NEXT:    s_xor_saveexec_b64 s[6:7], -1
; GCN-O0-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:4 ; 4-byte Folded Reload
; GCN-O0-NEXT:    s_mov_b64 exec, s[6:7]
; GCN-O0-NEXT:    s_add_i32 s32, s32, 0xfffff800
; GCN-O0-NEXT:    s_mov_b32 s33, s4
; GCN-O0-NEXT:    s_waitcnt vmcnt(0)
; GCN-O0-NEXT:    s_setpc_b64 s[30:31]
  call void @ext_func()
  ret void
}

declare void @ext_func();

attributes #0 = { nounwind "amdgpu-num-vgpr"="4" }
