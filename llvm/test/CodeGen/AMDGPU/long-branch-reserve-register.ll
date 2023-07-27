; RUN: llc -march=amdgcn -verify-machineinstrs -amdgpu-s-branch-bits=4 -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; OBJ:       Relocations [
; OBJ-NEXT: ]

; Used to emit an always 4 byte instruction. Inline asm always assumes
; each instruction is the maximum size.
declare void @llvm.amdgcn.s.sleep(i32) #0

declare i32 @llvm.amdgcn.workitem.id.x() #1


define amdgpu_kernel void @uniform_conditional_max_short_forward_branch(ptr addrspace(1) %arg, i32 %cnd) #0 {
; GCN-LABEL: uniform_conditional_max_short_forward_branch:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s2, s[0:1], 0xb
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_cmp_eq_u32 s2, 0
; GCN-NEXT:    s_cbranch_scc1 .LBB0_2
; GCN-NEXT:  ; %bb.1: ; %bb2
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:    s_sleep 0
; GCN-NEXT:  .LBB0_2: ; %bb3
; GCN-NEXT:    s_load_dwordx2 s[4:5], s[0:1], 0x9
; GCN-NEXT:    s_mov_b32 s7, 0xf000
; GCN-NEXT:    s_mov_b32 s6, -1
; GCN-NEXT:    v_mov_b32_e32 v0, s2
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_endpgm
bb:
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2 ; +8 dword branch

bb2:
; 24 bytes
  call void asm sideeffect
  "v_nop_e64
  v_nop_e64
  v_nop_e64", ""() #0
  call void @llvm.amdgcn.s.sleep(i32 0)
  br label %bb3

bb3:
  store volatile i32 %cnd, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @uniform_conditional_min_long_forward_branch(ptr addrspace(1) %arg, i32 %cnd) #0 {
; GCN-LABEL: uniform_conditional_min_long_forward_branch:
; GCN:       ; %bb.0: ; %bb0
; GCN-NEXT:  	 s_load_dword s2, s[0:1], 0xb
; GCN-NEXT:  	 s_waitcnt lgkmcnt(0)
; GCN-NEXT:  	 s_cmp_eq_u32 s2, 0
; GCN-NEXT:  	 s_cbranch_scc0 .LBB1_1
; GCN-NEXT:  .LBB1_3: ; %bb0
; GCN-NEXT:  	 s_getpc_b64 s[8:9]
; GCN-NEXT:  .Lpost_getpc0:
; GCN-NEXT:  	 s_add_u32 s8, s8, (.LBB1_2-.Lpost_getpc0)&4294967295
; GCN-NEXT:  	 s_addc_u32 s9, s9, (.LBB1_2-.Lpost_getpc0)>>32
; GCN-NEXT:  	 s_setpc_b64 s[8:9]
; GCN-NEXT:  .LBB1_1: ; %bb2
; GCN-NEXT:  	 ;;#ASMSTART
; GCN-NEXT:  	 v_nop_e64
; GCN-NEXT:     v_nop_e64
; GCN-NEXT:     v_nop_e64
; GCN-NEXT:     v_nop_e64
; GCN-NEXT:  	 ;;#ASMEND
; GCN-NEXT:  .LBB1_2: ; %bb3
; GCN-NEXT:  	 s_load_dwordx2 s[4:5], s[0:1], 0x9
; GCN-NEXT:  	 s_mov_b32 s7, 0xf000
; GCN-NEXT:  	 s_mov_b32 s6, -1
; GCN-NEXT:  	 v_mov_b32_e32 v0, s2
; GCN-NEXT:  	 s_waitcnt lgkmcnt(0)
; GCN-NEXT:  	 buffer_store_dword v0, off, s[4:7], 0
; GCN-NEXT:  	 s_waitcnt vmcnt(0)
; GCN-NEXT:  	 s_endpgm
bb0:
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2 ; +9 dword branch

bb2:
; 32 bytes
  call void asm sideeffect
  "v_nop_e64
  v_nop_e64
  v_nop_e64
  v_nop_e64", ""() #0
  br label %bb3

bb3:
  store volatile i32 %cnd, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @uniform_conditional_min_long_forward_vcnd_branch(ptr addrspace(1) %arg, float %cnd) #0 {
; GCN-LABEL: uniform_conditional_min_long_forward_vcnd_branch:
; GCN:       ; %bb.0: ; %bb0
; GCN-NEXT:    s_load_dword s2, s[0:1], 0xb
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_cmp_eq_f32_e64 s[4:5], s2, 0
; GCN-NEXT:    s_and_b64 vcc, exec, s[4:5]
; GCN-NEXT:    s_cbranch_vccz .LBB2_1
; GCN-NEXT:  .LBB2_3: ; %bb0
; GCN-NEXT:    s_getpc_b64 s[8:9]
; GCN-NEXT:  .Lpost_getpc1:
; GCN-NEXT:    s_add_u32 s8, s8, (.LBB2_2-.Lpost_getpc1)&4294967295
; GCN-NEXT:    s_addc_u32 s9, s9, (.LBB2_2-.Lpost_getpc1)>>32
; GCN-NEXT:    s_setpc_b64 s[8:9]
; GCN-NEXT:  .LBB2_1: ; %bb2
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:     ; 32 bytes
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:  .LBB2_2: ; %bb3
; GCN-NEXT:    s_load_dwordx2 s[4:5], s[0:1], 0x9
; GCN-NEXT:    s_mov_b32 s7, 0xf000
; GCN-NEXT:    s_mov_b32 s6, -1
; GCN-NEXT:    v_mov_b32_e32 v0, s2
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_endpgm
bb0:
  %cmp = fcmp oeq float %cnd, 0.0
  br i1 %cmp, label %bb3, label %bb2 ; + 8 dword branch

bb2:
  call void asm sideeffect " ; 32 bytes
  v_nop_e64
  v_nop_e64
  v_nop_e64
  v_nop_e64", ""() #0
  br label %bb3

bb3:
  store volatile float %cnd, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @min_long_forward_vbranch(ptr addrspace(1) %arg) #0 {
; GCN-LABEL: min_long_forward_vbranch:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; GCN-NEXT:    v_lshlrev_b32_e32 v0, 2, v0
; GCN-NEXT:    v_mov_b32_e32 v1, 0
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    s_mov_b32 s2, 0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    buffer_load_dword v2, v[0:1], s[0:3], 0 addr64 glc
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_mov_b32_e32 v1, s1
; GCN-NEXT:    v_add_i32_e32 v0, vcc, s0, v0
; GCN-NEXT:    v_addc_u32_e32 v1, vcc, 0, v1, vcc
; GCN-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v2
; GCN-NEXT:    s_and_saveexec_b64 s[0:1], vcc
; GCN-NEXT:    s_cbranch_execnz .LBB3_1
; GCN-NEXT:  .LBB3_3: ; %bb
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:  .Lpost_getpc2:
; GCN-NEXT:    s_add_u32 s4, s4, (.LBB3_2-.Lpost_getpc2)&4294967295
; GCN-NEXT:    s_addc_u32 s5, s5, (.LBB3_2-.Lpost_getpc2)>>32
; GCN-NEXT:    s_setpc_b64 s[4:5]
; GCN-NEXT:  .LBB3_1: ; %bb2
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:     ; 32 bytes
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:  .LBB3_2: ; %bb3
; GCN-NEXT:    s_or_b64 exec, exec, s[0:1]
; GCN-NEXT:    s_mov_b32 s0, s2
; GCN-NEXT:    s_mov_b32 s1, s2
; GCN-NEXT:    buffer_store_dword v2, v[0:1], s[0:3], 0 addr64
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_endpgm
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = zext i32 %tid to i64
  %gep = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 %tid.ext
  %load = load volatile i32, ptr addrspace(1) %gep
  %cmp = icmp eq i32 %load, 0
  br i1 %cmp, label %bb3, label %bb2 ; + 8 dword branch

bb2:
  call void asm sideeffect " ; 32 bytes
  v_nop_e64
  v_nop_e64
  v_nop_e64
  v_nop_e64", ""() #0
  br label %bb3

bb3:
  store volatile i32 %load, ptr addrspace(1) %gep
  ret void
}

define amdgpu_kernel void @long_backward_sbranch(ptr addrspace(1) %arg) #0 {
; GCN-LABEL: long_backward_sbranch:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_mov_b32 s0, 0
; GCN-NEXT:  .LBB4_1: ; %bb2
; GCN-NEXT:    ; =>This Inner Loop Header: Depth=1
; GCN-NEXT:    s_add_i32 s0, s0, 1
; GCN-NEXT:    s_cmp_lt_i32 s0, 10
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:    s_cbranch_scc0 .LBB4_2
; GCN-NEXT:  .LBB4_3: ; %bb2
; GCN-NEXT:    ; in Loop: Header=BB4_1 Depth=1
; GCN-NEXT:    s_getpc_b64 s[2:3]
; GCN-NEXT:  .Lpost_getpc3:
; GCN-NEXT:    s_add_u32 s2, s2, (.LBB4_1-.Lpost_getpc3)&4294967295
; GCN-NEXT:    s_addc_u32 s3, s3, (.LBB4_1-.Lpost_getpc3)>>32
; GCN-NEXT:    s_setpc_b64 s[2:3]
; GCN-NEXT:  .LBB4_2: ; %bb3
; GCN-NEXT:    s_endpgm

bb:
  br label %bb2

bb2:
  %loop.idx = phi i32 [ 0, %bb ], [ %inc, %bb2 ]
  ; 24 bytes
  call void asm sideeffect
  "v_nop_e64
  v_nop_e64
  v_nop_e64", ""() #0
  %inc = add nsw i32 %loop.idx, 1 ; add cost 4
  %cmp = icmp slt i32 %inc, 10 ; condition cost = 8
  br i1 %cmp, label %bb2, label %bb3 ; -

bb3:
  ret void
}

; Requires expansion of unconditional branch from %bb2 to %bb4 (and
; expansion of conditional branch from %bb to %bb3.

define amdgpu_kernel void @uniform_unconditional_min_long_forward_branch(ptr addrspace(1) %arg, i32 %arg1) {
; GCN-LABEL: uniform_unconditional_min_long_forward_branch:
; GCN:       ; %bb.0: ; %bb0
; GCN-NEXT:    s_load_dword s2, s[0:1], 0xb
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_cmp_eq_u32 s2, 0
; GCN-NEXT:    s_mov_b64 s[2:3], -1
; GCN-NEXT:    s_cbranch_scc0 .LBB5_1
; GCN-NEXT:  .LBB5_7: ; %bb0
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:  .Lpost_getpc5:
; GCN-NEXT:    s_add_u32 s4, s4, (.LBB5_4-.Lpost_getpc5)&4294967295
; GCN-NEXT:    s_addc_u32 s5, s5, (.LBB5_4-.Lpost_getpc5)>>32
; GCN-NEXT:    s_setpc_b64 s[4:5]
; GCN-NEXT:  .LBB5_1: ; %Flow
; GCN-NEXT:    s_andn2_b64 vcc, exec, s[2:3]
; GCN-NEXT:    s_cbranch_vccnz .LBB5_3
; GCN-NEXT:  .LBB5_2: ; %bb2
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    s_mov_b32 s2, -1
; GCN-NEXT:    v_mov_b32_e32 v0, 17
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:  .LBB5_3: ; %bb4
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    s_mov_b32 s2, -1
; GCN-NEXT:    s_waitcnt expcnt(0)
; GCN-NEXT:    v_mov_b32_e32 v0, 63
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_endpgm
; GCN-NEXT:  .LBB5_4: ; %bb3
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    v_nop_e64
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:  s_mov_b64 vcc, exec
; GCN-NEXT:    s_cbranch_execnz .LBB5_5
; GCN-NEXT:  .LBB5_9: ; %bb3
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:  .Lpost_getpc6:
; GCN-NEXT:    s_add_u32 s4, s4, (.LBB5_2-.Lpost_getpc6)&4294967295
; GCN-NEXT:    s_addc_u32 s5, s5, (.LBB5_2-.Lpost_getpc6)>>32
; GCN-NEXT:    s_setpc_b64 s[4:5]
; GCN-NEXT:  .LBB5_5: ; %bb3
; GCN-NEXT:    s_getpc_b64 s[4:5]
; GCN-NEXT:  .Lpost_getpc4:
; GCN-NEXT:    s_add_u32 s4, s4, (.LBB5_3-.Lpost_getpc4)&4294967295
; GCN-NEXT:    s_addc_u32 s5, s5, (.LBB5_3-.Lpost_getpc4)>>32
; GCN-NEXT:    s_setpc_b64 s[4:5]
bb0:
  %tmp = icmp ne i32 %arg1, 0
  br i1 %tmp, label %bb2, label %bb3

bb2:
  store volatile i32 17, ptr addrspace(1) undef
  br label %bb4

bb3:
  ; 32 byte asm
  call void asm sideeffect
  "v_nop_e64
  v_nop_e64
  v_nop_e64
  v_nop_e64", ""() #0
  br label %bb4

bb4:
  store volatile i32 63, ptr addrspace(1) %arg
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
