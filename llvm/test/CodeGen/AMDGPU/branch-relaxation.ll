; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -amdgpu-s-branch-bits=4 < %s | FileCheck -enable-var-scope -check-prefix=GCN %s


; FIXME: We should use llvm-mc for this, but we can't even parse our own output.
;        See PR33579.
; RUN: llc -march=amdgcn -verify-machineinstrs -amdgpu-s-branch-bits=4 -o %t.o -filetype=obj %s
; RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=OBJ %s

; OBJ:       Relocations [
; OBJ-NEXT: ]

; Restrict maximum branch to between +7 and -8 dwords

; Used to emit an always 4 byte instruction. Inline asm always assumes
; each instruction is the maximum size.
declare void @llvm.amdgcn.s.sleep(i32) #0

declare i32 @llvm.amdgcn.workitem.id.x() #1


; GCN-LABEL: {{^}}uniform_conditional_max_short_forward_branch:
; GCN: s_load_dword [[CND:s[0-9]+]]
; GCN: s_cmp_eq_u32 [[CND]], 0
; GCN-NEXT: s_cbranch_scc1 [[BB3:BB[0-9]+_[0-9]+]]


; GCN-NEXT: ; %bb.1: ; %bb2
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_sleep 0

; GCN-NEXT: [[BB3]]: ; %bb3
; GCN: v_mov_b32_e32 [[V_CND:v[0-9]+]], [[CND]]
; GCN: buffer_store_dword [[V_CND]]
; GCN: s_endpgm
define amdgpu_kernel void @uniform_conditional_max_short_forward_branch(i32 addrspace(1)* %arg, i32 %cnd) #0 {
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
  store volatile i32 %cnd, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}uniform_conditional_min_long_forward_branch:
; GCN: s_load_dword [[CND:s[0-9]+]]
; GCN: s_cmp_eq_u32 [[CND]], 0
; GCN-NEXT: s_cbranch_scc0 [[LONGBB:BB[0-9]+_[0-9]+]]

; GCN-NEXT: [[LONG_JUMP:BB[0-9]+_[0-9]+]]: ; %bb0
; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], [[ENDBB:BB[0-9]+_[0-9]+]]-([[LONG_JUMP]]+4)
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], 0
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}

; GCN-NEXT: [[LONGBB]]:
; GCN-NEXT: ;;#ASMSTART
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN-NEXT: ;;#ASMEND

; GCN-NEXT: [[ENDBB]]:
; GCN: v_mov_b32_e32 [[V_CND:v[0-9]+]], [[CND]]
; GCN: buffer_store_dword [[V_CND]]
; GCN: s_endpgm
define amdgpu_kernel void @uniform_conditional_min_long_forward_branch(i32 addrspace(1)* %arg, i32 %cnd) #0 {
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
  store volatile i32 %cnd, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}uniform_conditional_min_long_forward_vcnd_branch:
; GCN: s_load_dword [[CND:s[0-9]+]]

; GCN-DAG: v_cmp_eq_f32_e64 [[UNMASKED:s\[[0-9]+:[0-9]+\]]], [[CND]], 0
; GCN-DAG: s_and_b64 vcc, exec, [[UNMASKED]]
; GCN: s_cbranch_vccz [[LONGBB:BB[0-9]+_[0-9]+]]

; GCN-NEXT: [[LONG_JUMP:BB[0-9]+_[0-9]+]]: ; %bb0
; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], [[ENDBB:BB[0-9]+_[0-9]+]]-([[LONG_JUMP]]+4)
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], 0
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}

; GCN-NEXT: [[LONGBB]]:
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64

; GCN: [[ENDBB]]:
; GCN: v_mov_b32_e32 [[V_CND:v[0-9]+]], [[CND]]
; GCN: buffer_store_dword [[V_CND]]
; GCN: s_endpgm
define amdgpu_kernel void @uniform_conditional_min_long_forward_vcnd_branch(float addrspace(1)* %arg, float %cnd) #0 {
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
  store volatile float %cnd, float addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}min_long_forward_vbranch:

; GCN: buffer_load_dword
; GCN: v_cmp_ne_u32_e32 vcc, 0, v{{[0-9]+}}
; GCN: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], vcc

; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64

; GCN: s_or_b64 exec, exec, [[SAVE]]
; GCN: buffer_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @min_long_forward_vbranch(i32 addrspace(1)* %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = zext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tid.ext
  %load = load volatile i32, i32 addrspace(1)* %gep
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
  store volatile i32 %load, i32 addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}long_backward_sbranch:
; GCN: s_mov_b32 [[LOOPIDX:s[0-9]+]], 0{{$}}

; GCN: [[LOOPBB:BB[0-9]+_[0-9]+]]: ; %bb2
; GCN-NEXT: ; =>This Inner Loop Header: Depth=1
; GCN-NEXT: s_add_i32 [[INC:s[0-9]+]], [[LOOPIDX]], 1
; GCN-NEXT: s_cmp_lt_i32 [[INC]], 10

; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: ;;#ASMEND

; GCN-NEXT: s_cbranch_scc0 [[ENDBB:BB[0-9]+_[0-9]+]]

; GCN-NEXT: [[LONG_JUMP:BB[0-9]+_[0-9]+]]: ; %bb2
; GCN-NEXT: ; in Loop: Header=[[LOOPBB]] Depth=1

; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_sub_u32 s[[PC_LO]], s[[PC_LO]], ([[LONG_JUMP]]+4)-[[LOOPBB]]
; GCN-NEXT: s_subb_u32 s[[PC_HI]], s[[PC_HI]], 0
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}

; GCN-NEXT: [[ENDBB]]:
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @long_backward_sbranch(i32 addrspace(1)* %arg) #0 {
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

; GCN-LABEL: {{^}}uniform_unconditional_min_long_forward_branch:
; GCN: s_cmp_eq_u32
; GCN-NEXT: s_cbranch_scc0 [[BB2:BB[0-9]+_[0-9]+]]

; GCN-NEXT: [[LONG_JUMP0:BB[0-9]+_[0-9]+]]: ; %bb0
; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC0_LO:[0-9]+]]:[[PC0_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC0_LO]], s[[PC0_LO]], [[BB3:BB[0-9]_[0-9]+]]-([[LONG_JUMP0]]+4)
; GCN-NEXT: s_addc_u32 s[[PC0_HI]], s[[PC0_HI]], 0{{$}}
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC0_LO]]:[[PC0_HI]]{{\]}}

; GCN-NEXT: [[BB2]]: ; %bb2
; GCN: v_mov_b32_e32 [[BB2_K:v[0-9]+]], 17
; GCN: buffer_store_dword [[BB2_K]]

; GCN-NEXT: [[LONG_JUMP1:BB[0-9]+_[0-9]+]]: ; %bb2
; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC1_LO:[0-9]+]]:[[PC1_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC1_LO]], s[[PC1_LO]], [[BB4:BB[0-9]_[0-9]+]]-([[LONG_JUMP1]]+4)
; GCN-NEXT: s_addc_u32 s[[PC1_HI]], s[[PC1_HI]], 0{{$}}
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC1_LO]]:[[PC1_HI]]{{\]}}

; GCN: [[BB3]]: ; %bb3
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: ;;#ASMEND

; GCN-NEXT: [[BB4]]: ; %bb4
; GCN: v_mov_b32_e32 [[BB4_K:v[0-9]+]], 63
; GCN: buffer_store_dword [[BB4_K]]
; GCN-NEXT: s_endpgm
; GCN-NEXT: .Lfunc_end{{[0-9]+}}:
define amdgpu_kernel void @uniform_unconditional_min_long_forward_branch(i32 addrspace(1)* %arg, i32 %arg1) {
bb0:
  %tmp = icmp ne i32 %arg1, 0
  br i1 %tmp, label %bb2, label %bb3

bb2:
  store volatile i32 17, i32 addrspace(1)* undef
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
  store volatile i32 63, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}uniform_unconditional_min_long_backward_branch:
; GCN-NEXT: ; %bb.0: ; %entry

; GCN-NEXT: [[LOOP:BB[0-9]_[0-9]+]]: ; %loop
; GCN-NEXT: ; =>This Inner Loop Header: Depth=1
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: ;;#ASMEND

; GCN-NEXT: [[LONGBB:BB[0-9]+_[0-9]+]]: ; %loop
; GCN-NEXT: ; in Loop: Header=[[LOOP]] Depth=1

; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_sub_u32 s[[PC_LO]], s[[PC_LO]], ([[LONGBB]]+4)-[[LOOP]]
; GCN-NEXT: s_subb_u32 s[[PC_HI]], s[[PC_HI]], 0{{$}}
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT .Lfunc_end{{[0-9]+}}:
define amdgpu_kernel void @uniform_unconditional_min_long_backward_branch(i32 addrspace(1)* %arg, i32 %arg1) {
entry:
  br label %loop

loop:
  ; 32 byte asm
  call void asm sideeffect
   "v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64", ""() #0
  br label %loop
}

; Expansion of branch from %bb1 to %bb3 introduces need to expand
; branch from %bb0 to %bb2

; GCN-LABEL: {{^}}expand_requires_expand:
; GCN-NEXT: ; %bb.0: ; %bb0
; GCN: s_load_dword
; GCN: s_cmp_lt_i32 s{{[0-9]+}}, 0{{$}}
; GCN-NEXT: s_cbranch_scc0 [[BB1:BB[0-9]+_[0-9]+]]

; GCN-NEXT: [[LONGBB0:BB[0-9]+_[0-9]+]]: ; %bb0

; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC0_LO:[0-9]+]]:[[PC0_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC0_LO]], s[[PC0_LO]], [[BB2:BB[0-9]_[0-9]+]]-([[LONGBB0]]+4)
; GCN-NEXT: s_addc_u32 s[[PC0_HI]], s[[PC0_HI]], 0{{$}}
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC0_LO]]:[[PC0_HI]]{{\]}}

; GCN-NEXT: [[BB1]]: ; %bb1
; GCN-NEXT: s_load_dword
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: s_cmp_eq_u32 s{{[0-9]+}}, 3{{$}}
; GCN-NEXT: s_cbranch_scc0 [[BB2:BB[0-9]_[0-9]+]]

; GCN-NEXT: [[LONGBB1:BB[0-9]+_[0-9]+]]: ; %bb1
; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC1_LO:[0-9]+]]:[[PC1_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC1_LO]], s[[PC1_LO]], [[BB3:BB[0-9]+_[0-9]+]]-([[LONGBB1]]+4)
; GCN-NEXT: s_addc_u32 s[[PC1_HI]], s[[PC1_HI]], 0{{$}}
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC1_LO]]:[[PC1_HI]]{{\]}}

; GCN-NEXT: [[BB2]]: ; %bb2
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: v_nop_e64
; GCN-NEXT: ;;#ASMEND

; GCN-NEXT: [[BB3]]: ; %bb3
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: v_nop_e64
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: v_nop_e64
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @expand_requires_expand(i32 %cond0) #0 {
bb0:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %cmp0 = icmp slt i32 %cond0, 0
  br i1 %cmp0, label %bb2, label %bb1

bb1:
  %val = load volatile i32, i32 addrspace(4)* undef
  %cmp1 = icmp eq i32 %val, 3
  br i1 %cmp1, label %bb3, label %bb2

bb2:
  call void asm sideeffect
   "v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64", ""() #0
  br label %bb3

bb3:
; These NOPs prevent tail-duplication-based outlining
; from firing, which defeats the need to expand the branches and this test.
  call void asm sideeffect
   "v_nop_e64", ""() #0
  call void asm sideeffect
   "v_nop_e64", ""() #0
  ret void
}

; Requires expanding of required skip branch.

; GCN-LABEL: {{^}}uniform_inside_divergent:
; GCN: v_cmp_gt_u32_e32 vcc, 16, v{{[0-9]+}}
; GCN-NEXT: s_and_saveexec_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], vcc
; GCN-NEXT: ; mask branch [[ENDIF:BB[0-9]+_[0-9]+]]
; GCN-NEXT: s_cbranch_execnz [[IF:BB[0-9]+_[0-9]+]]

; GCN-NEXT: [[LONGBB:BB[0-9]+_[0-9]+]]: ; %entry
; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], [[BB2:BB[0-9]_[0-9]+]]-([[LONGBB]]+4)
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], 0{{$}}
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}

; GCN-NEXT: [[IF]]: ; %if
; GCN: buffer_store_dword
; GCN: s_cmp_lg_u32
; GCN: s_cbranch_scc1 [[ENDIF]]

; GCN-NEXT: ; %bb.2: ; %if_uniform
; GCN: buffer_store_dword

; GCN-NEXT: [[ENDIF]]: ; %endif
; GCN-NEXT: s_or_b64 exec, exec, [[MASK]]
; GCN-NEXT: s_sleep 5
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @uniform_inside_divergent(i32 addrspace(1)* %out, i32 %cond) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %d_cmp = icmp ult i32 %tid, 16
  br i1 %d_cmp, label %if, label %endif

if:
  store i32 0, i32 addrspace(1)* %out
  %u_cmp = icmp eq i32 %cond, 0
  br i1 %u_cmp, label %if_uniform, label %endif

if_uniform:
  store i32 1, i32 addrspace(1)* %out
  br label %endif

endif:
  ; layout can remove the split branch if it can copy the return block.
  ; This call makes the return block long enough that it doesn't get copied.
  call void @llvm.amdgcn.s.sleep(i32 5);
  ret void
}

; si_mask_branch

; GCN-LABEL: {{^}}analyze_mask_branch:
; GCN: v_cmp_nlt_f32_e32 vcc
; GCN-NEXT: s_and_saveexec_b64 [[TEMP_MASK:s\[[0-9]+:[0-9]+\]]], vcc
; GCN-NEXT: s_xor_b64  [[MASK:s\[[0-9]+:[0-9]+\]]], exec, [[TEMP_MASK]]
; GCN-NEXT: ; mask branch [[FLOW:BB[0-9]+_[0-9]+]]

; GCN: [[FLOW]]: ; %Flow
; GCN-NEXT: s_or_saveexec_b64 [[TEMP_MASK1:s\[[0-9]+:[0-9]+\]]], [[MASK]]
; GCN-NEXT: s_xor_b64 exec, exec, [[TEMP_MASK1]]
; GCN-NEXT: ; mask branch [[RET:BB[0-9]+_[0-9]+]]

; GCN: [[LOOP_BODY:BB[0-9]+_[0-9]+]]: ; %loop{{$}}
; GCN: ;;#ASMSTART
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: v_nop_e64
; GCN: ;;#ASMEND
; GCN: s_cbranch_vccz [[RET]]

; GCN-NEXT: [[LONGBB:BB[0-9]+_[0-9]+]]: ; %loop
; GCN-NEXT: ; in Loop: Header=[[LOOP_BODY]] Depth=1
; GCN-NEXT: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_sub_u32 s[[PC_LO]], s[[PC_LO]], ([[LONGBB]]+4)-[[LOOP_BODY]]
; GCN-NEXT: s_subb_u32 s[[PC_HI]], s[[PC_HI]], 0
; GCN-NEXT: s_setpc_b64 s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}

; GCN-NEXT: [[RET]]: ; %UnifiedReturnBlock
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @analyze_mask_branch() #0 {
entry:
  %reg = call float asm sideeffect "v_mov_b32_e64 $0, 0", "=v"()
  %cmp0 = fcmp ogt float %reg, 0.000000e+00
  br i1 %cmp0, label %loop, label %ret

loop:
  %phi = phi float [ 0.000000e+00, %loop_body ], [ 1.000000e+00, %entry ]
  call void asm sideeffect
    "v_nop_e64
     v_nop_e64", ""() #0
  %cmp1 = fcmp olt float %phi, 8.0
  br i1 %cmp1, label %loop_body, label %ret

loop_body:
  call void asm sideeffect
  "v_nop_e64
   v_nop_e64
   v_nop_e64
   v_nop_e64", ""() #0
  br label %loop

ret:
  store volatile i32 7, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}long_branch_hang:
; GCN: s_cmp_lt_i32 s{{[0-9]+}}, 6
; GCN: s_cbranch_scc1 {{BB[0-9]+_[0-9]+}}
; GCN-NEXT: s_branch [[LONG_BR_0:BB[0-9]+_[0-9]+]]
; GCN-NEXT: BB{{[0-9]+_[0-9]+}}:

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, [[LONG_BR_DEST0:BB[0-9]+_[0-9]+]]-(
; GCN-NEXT: s_addc_u32
; GCN-NEXT: s_setpc_b64

; GCN-NEXT: [[LONG_BR_0]]:
; GCN-DAG: v_cmp_lt_i32
; GCN-DAG: v_cmp_gt_i32
; GCN: s_cbranch_vccnz

; GCN: s_setpc_b64
; GCN: s_setpc_b64

; GCN: [[LONG_BR_DEST0]]
; GCN: s_cbranch_vccz
; GCN: s_setpc_b64

; GCN: s_endpgm
define amdgpu_kernel void @long_branch_hang(i32 addrspace(1)* nocapture %arg, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4, i64 %arg5) #0 {
bb:
  %tmp = icmp slt i32 %arg2, 9
  %tmp6 = icmp eq i32 %arg1, 0
  %tmp7 = icmp sgt i32 %arg4, 0
  %tmp8 = icmp sgt i32 %arg4, 5
  br i1 %tmp8, label %bb9, label %bb13

bb9:                                              ; preds = %bb
  %tmp10 = and i1 %tmp7, %tmp
  %tmp11 = icmp slt i32 %arg3, %arg4
  %tmp12 = or i1 %tmp11, %tmp7
  br i1 %tmp12, label %bb19, label %bb14

bb13:                                             ; preds = %bb
  call void asm sideeffect
  "v_nop_e64
   v_nop_e64
   v_nop_e64
   v_nop_e64", ""() #0
  br i1 %tmp6, label %bb19, label %bb14

bb14:                                             ; preds = %bb13, %bb9
  %tmp15 = icmp slt i32 %arg3, %arg4
  %tmp16 = or i1 %tmp15, %tmp
  %tmp17 = and i1 %tmp6, %tmp16
  %tmp18 = zext i1 %tmp17 to i32
  br label %bb19

bb19:                                             ; preds = %bb14, %bb13, %bb9
  %tmp20 = phi i32 [ undef, %bb9 ], [ undef, %bb13 ], [ %tmp18, %bb14 ]
  %tmp21 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %arg5
  store i32 %tmp20, i32 addrspace(1)* %tmp21, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
