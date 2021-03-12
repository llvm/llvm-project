; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Check that we properly realign the stack. While 4-byte access is all
; that is ever needed, some transformations rely on the known bits from the alignment of the pointer (e.g.


; 128 byte object
; 4 byte emergency stack slot
; = 144 bytes with padding between them

; GCN-LABEL: {{^}}needs_align16_default_stack_align:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 4, v0
; GCN-DAG: v_lshrrev_b32_e64 [[FRAMEDIFF:v[0-9]+]], 6, s32
; GCN: v_add_u32_e32 [[FI:v[0-9]+]], vcc, [[FRAMEDIFF]], [[SCALED_IDX]]

; GCN-NOT: s32

; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: v_or_b32_e32 v{{[0-9]+}}, 12
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen

; GCN-NOT: s32

; GCN: ; ScratchSize: 144
define void @needs_align16_default_stack_align(i32 %idx) #0 {
  %alloca.align16 = alloca [8 x <4 x i32>], align 16, addrspace(5)
  %gep0 = getelementptr inbounds [8 x <4 x i32>], [8 x <4 x i32>] addrspace(5)* %alloca.align16, i32 0, i32 %idx
  store volatile <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> addrspace(5)* %gep0, align 16
  ret void
}

; GCN-LABEL: {{^}}needs_align16_stack_align4:
; GCN: s_add_u32 [[SCRATCH_REG:s[0-9]+]], s32, 0x3c0{{$}}
; GCN: s_and_b32 s33, [[SCRATCH_REG]], 0xfffffc00

; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: v_or_b32_e32 v{{[0-9]+}}, 12
; GCN: s_add_u32 s32, s32, 0x2800{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen

; GCN: s_sub_u32 s32, s32, 0x2800

; GCN: ; ScratchSize: 160
define void @needs_align16_stack_align4(i32 %idx) #2 {
  %alloca.align16 = alloca [8 x <4 x i32>], align 16, addrspace(5)
  %gep0 = getelementptr inbounds [8 x <4 x i32>], [8 x <4 x i32>] addrspace(5)* %alloca.align16, i32 0, i32 %idx
  store volatile <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> addrspace(5)* %gep0, align 16
  ret void
}

; GCN-LABEL: {{^}}needs_align32:
; GCN: s_add_u32 [[SCRATCH_REG:s[0-9]+]], s32, 0x7c0{{$}}
; GCN: s_and_b32 s33, [[SCRATCH_REG]], 0xfffff800

; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: v_or_b32_e32 v{{[0-9]+}}, 12
; GCN: s_add_u32 s32, s32, 0x3000{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen

; GCN: s_sub_u32 s32, s32, 0x3000

; GCN: ; ScratchSize: 192
define void @needs_align32(i32 %idx) #0 {
  %alloca.align16 = alloca [8 x <4 x i32>], align 32, addrspace(5)
  %gep0 = getelementptr inbounds [8 x <4 x i32>], [8 x <4 x i32>] addrspace(5)* %alloca.align16, i32 0, i32 %idx
  store volatile <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> addrspace(5)* %gep0, align 32
  ret void
}

; GCN-LABEL: {{^}}force_realign4:
; GCN: s_add_u32 [[SCRATCH_REG:s[0-9]+]], s32, 0xc0{{$}}
; GCN: s_and_b32 s33, [[SCRATCH_REG]], 0xffffff00
; GCN: s_add_u32 s32, s32, 0xd00{{$}}

; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], 0 offen
; GCN: s_sub_u32 s32, s32, 0xd00

; GCN: ; ScratchSize: 52
define void @force_realign4(i32 %idx) #1 {
  %alloca.align16 = alloca [8 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [8 x i32], [8 x i32] addrspace(5)* %alloca.align16, i32 0, i32 %idx
  store volatile i32 3, i32 addrspace(5)* %gep0, align 4
  ret void
}

; GCN-LABEL: {{^}}kernel_call_align16_from_8:
; GCN: s_movk_i32 s32, 0x400{{$}}
; GCN-NOT: s32
; GCN: s_swappc_b64
define amdgpu_kernel void @kernel_call_align16_from_8() #0 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 2, i32 addrspace(5)* %alloca
  call void @needs_align16_default_stack_align(i32 1)
  ret void
}

; The call sequence should keep the stack on call aligned to 4
; GCN-LABEL: {{^}}kernel_call_align16_from_5:
; GCN: s_movk_i32 s32, 0x400
; GCN: s_swappc_b64
define amdgpu_kernel void @kernel_call_align16_from_5() {
  %alloca0 = alloca i8, align 1, addrspace(5)
  store volatile i8 2, i8  addrspace(5)* %alloca0

  call void @needs_align16_default_stack_align(i32 1)
  ret void
}

; GCN-LABEL: {{^}}kernel_call_align4_from_5:
; GCN: s_movk_i32 s32, 0x400
; GCN: s_swappc_b64
define amdgpu_kernel void @kernel_call_align4_from_5() {
  %alloca0 = alloca i8, align 1, addrspace(5)
  store volatile i8 2, i8  addrspace(5)* %alloca0

  call void @needs_align16_stack_align4(i32 1)
  ret void
}

; GCN-LABEL: {{^}}default_realign_align128:
; GCN: s_mov_b32 [[FP_COPY:s[0-9]+]], s33
; GCN-NEXT: s_add_u32 s33, s32, 0x1fc0
; GCN-NEXT: s_and_b32 s33, s33, 0xffffe000
; GCN-NEXT: s_add_u32 s32, s32, 0x4000
; GCN-NOT: s33
; GCN: buffer_store_dword v0, off, s[0:3], s33{{$}}
; GCN: s_sub_u32 s32, s32, 0x4000
; GCN: s_mov_b32 s33, [[FP_COPY]]
define void @default_realign_align128(i32 %idx) #0 {
  %alloca.align = alloca i32, align 128, addrspace(5)
  store volatile i32 9, i32 addrspace(5)* %alloca.align, align 128
  ret void
}

; GCN-LABEL: {{^}}disable_realign_align128:
; GCN-NOT: s32
; GCN: buffer_store_dword v0, off, s[0:3], s32{{$}}
; GCN-NOT: s32
define void @disable_realign_align128(i32 %idx) #3 {
  %alloca.align = alloca i32, align 128, addrspace(5)
  store volatile i32 9, i32 addrspace(5)* %alloca.align, align 128
  ret void
}

declare void @extern_func(<32 x i32>, i32) #0
define void @func_call_align1024_bp_gets_vgpr_spill(<32 x i32> %a, i32 %b) #0 {
; The test forces the stack to be realigned to a new boundary
; since there is a local object with an alignment of 1024.
; Should use BP to access the incoming stack arguments.
; The BP value is saved/restored with a VGPR spill.

; GCN-LABEL: func_call_align1024_bp_gets_vgpr_spill:
; GCN: buffer_store_dword [[VGPR_REG:v[0-9]+]], off, s[0:3], s32 offset:1028 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, s[16:17]
; GCN-NEXT: v_writelane_b32 [[VGPR_REG]], s33, 2
; GCN-DAG: s_add_u32 [[SCRATCH_REG:s[0-9]+]], s32, 0xffc0
; GCN: s_and_b32 s33, [[SCRATCH_REG]], 0xffff0000
; GCN: v_mov_b32_e32 v32, 0
; GCN-DAG: v_writelane_b32 [[VGPR_REG]], s34, 3
; GCN: s_mov_b32 s34, s32
; GCN: buffer_store_dword v32, off, s[0:3], s33 offset:1024
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s34
; GCN-DAG: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s34 offset:4
; GCN-DAG: s_add_u32 s32, s32, 0x30000
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32
; GCN-DAG: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4
; GCN-NEXT: s_swappc_b64 s[30:31], s[16:17]

; GCN: s_sub_u32 s32, s32, 0x30000
; GCN-NEXT: v_readlane_b32 s33, [[VGPR_REG]], 2
; GCN-NEXT: v_readlane_b32 s34, [[VGPR_REG]], 3
; GCN-NEXT: s_or_saveexec_b64 s[6:7], -1
; GCN-NEXT: buffer_load_dword [[VGPR_REG]], off, s[0:3], s32 offset:1028 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, s[6:7]
  %temp = alloca i32, align 1024, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %temp, align 1024
  call void @extern_func(<32 x i32> %a, i32 %b)
  ret void
}

%struct.Data = type { [9 x i32] }
define i32 @needs_align1024_stack_args_used_inside_loop(%struct.Data addrspace(5)* nocapture readonly byval(%struct.Data) align 8 %arg) local_unnamed_addr #4 {
; The local object allocation needed an alignment of 1024.
; Since the function argument is accessed in a loop with an
; index variable, the base pointer first get loaded into a VGPR
; and that value should be further referenced to load the incoming values.
; The BP value will get saved/restored in an SGPR at the prolgoue/epilogue.

; GCN-LABEL: needs_align1024_stack_args_used_inside_loop:
; GCN: s_mov_b32 [[FP_COPY:s[0-9]+]], s33
; GCN-NEXT: s_add_u32 s33, s32, 0xffc0
; GCN-NEXT: s_mov_b32 [[BP_COPY:s[0-9]+]], s34
; GCN-NEXT: s_mov_b32 s34, s32
; GCN-NEXT: s_and_b32 s33, s33, 0xffff0000
; GCN-NEXT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NEXT: v_lshrrev_b32_e64 [[VGPR_REG:v[0-9]+]], 6, s34
; GCN: s_add_u32 s32, s32, 0x30000
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33 offset:1024
; GCN: buffer_load_dword v{{[0-9]+}}, [[VGPR_REG]], s[0:3], 0 offen
; GCN: v_add_u32_e32 [[VGPR_REG]], vcc, 4, [[VGPR_REG]]
; GCN: s_sub_u32 s32, s32, 0x30000
; GCN-NEXT: s_mov_b32 s33, [[FP_COPY]]
; GCN-NEXT: s_mov_b32 s34, [[BP_COPY]]
; GCN-NEXT: s_setpc_b64 s[30:31]
begin:
  %local_var = alloca i32, align 1024, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %local_var, align 1024
  br label %loop_body

loop_end:                                                ; preds = %loop_body
  %idx_next = add nuw nsw i32 %lp_idx, 1
  %lp_exit_cond = icmp eq i32 %idx_next, 9
  br i1 %lp_exit_cond, label %exit, label %loop_body

loop_body:                                                ; preds = %loop_end, %begin
  %lp_idx = phi i32 [ 0, %begin ], [ %idx_next, %loop_end ]
  %ptr = getelementptr inbounds %struct.Data, %struct.Data addrspace(5)* %arg, i32 0, i32 0, i32 %lp_idx
  %val = load i32, i32 addrspace(5)* %ptr, align 8
  %lp_cond = icmp eq i32 %val, %lp_idx
  br i1 %lp_cond, label %loop_end, label %exit

exit:                                               ; preds = %loop_end, %loop_body
  %out = phi i32 [ 0, %loop_body ], [ 1, %loop_end ]
  ret i32 %out
}

define void @no_free_scratch_sgpr_for_bp_copy(<32 x i32> %a, i32 %b) #0 {
; GCN-LABEL: no_free_scratch_sgpr_for_bp_copy:
; GCN: ; %bb.0:
; GCN: v_writelane_b32 [[VGPR_REG:v[0-9]+]], s34, 0
; GCN-NEXT: s_mov_b32 s34, s32
; GCN-NEXT: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s34
; GCN: v_readlane_b32 s34, [[VGPR_REG:v[0-9]+]], 0
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33 offset:128
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; GCN: s_setpc_b64 s[30:31]
  %local_val = alloca i32, align 128, addrspace(5)
  store volatile i32 %b, i32 addrspace(5)* %local_val, align 128
  ; Use all clobberable registers, so BP has to spill to a VGPR.
  call void asm sideeffect "",
    "~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{vcc_hi}"() #0
  ret void
}

define void @no_free_regs_spill_bp_to_memory(<32 x i32> %a, i32 %b) #5 {
; If there are no free SGPRs or VGPRs available we must spill the BP to memory.

; GCN-LABEL: no_free_regs_spill_bp_to_mem
; GCN: s_or_saveexec_b64 s[4:5], -1
; GCN: v_mov_b32_e32 v0, s33
; GCN: buffer_store_dword v0, off, s[0:3], s32
; GCN: v_mov_b32_e32 v0, s34
; GCN-DAG: buffer_store_dword v0, off, s[0:3], s32
  %local_val = alloca i32, align 128, addrspace(5)
  store volatile i32 %b, i32 addrspace(5)* %local_val, align 128

  call void asm sideeffect "; clobber nonpreserved SGPRs and 64 CSRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102},~{s39},~{vcc}"() #0

  call void asm sideeffect "; clobber all VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38}" () #0
  ret void
}

define void @spill_bp_to_memory_scratch_reg_needed_mubuf_offset(<32 x i32> %a, i32 %b, [4096 x i8] addrspace(5)* byval([4096 x i8]) align 4 %arg) #5 {
; If the size of the offset exceeds the MUBUF offset field we need another
; scratch VGPR to hold the offset.

; GCN-LABEL: spill_bp_to_memory_scratch_reg_needed_mubuf_offset
; GCN: s_or_saveexec_b64 s[4:5], -1
; GCN: v_mov_b32_e32 v0, s33
; GCN-NOT: v_mov_b32_e32 v0, 0x1088
; GCN-NEXT: v_mov_b32_e32 v1, 0x1088
; GCN-NEXT: buffer_store_dword v0, v1, s[0:3], s32 offen
; GCN: v_mov_b32_e32 v0, s34
; GCN-NOT: v_mov_b32_e32 v0, 0x108c
; GCN-NEXT: v_mov_b32_e32 v1, 0x108c
; GCN-NEXT: buffer_store_dword v0, v1, s[0:3], s32 offen
  %local_val = alloca i32, align 128, addrspace(5)
  store volatile i32 %b, i32 addrspace(5)* %local_val, align 128

  call void asm sideeffect "; clobber nonpreserved SGPRs and 64 CSRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102},~{s39},~{vcc}"() #0

  call void asm sideeffect "; clobber all VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38}"() #0
  ret void
}

attributes #0 = { noinline nounwind }
attributes #1 = { noinline nounwind "stackrealign" }
attributes #2 = { noinline nounwind alignstack=4 }
attributes #3 = { noinline nounwind "no-realign-stack" }
attributes #4 = { noinline nounwind "frame-pointer"="all"}
attributes #5 = { noinline nounwind "amdgpu-waves-per-eu"="6,6" }
