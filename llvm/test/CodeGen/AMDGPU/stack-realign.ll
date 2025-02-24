; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Check that we properly realign the stack. While 4-byte access is all
; that is ever needed, some transformations rely on the known bits from the alignment of the pointer (e.g.


; 128 byte object
; 4 byte emergency stack slot
; = 144 bytes with padding between them

define void @needs_align16_default_stack_align(i32 %idx) #0 {
; GCN-LABEL: needs_align16_default_stack_align:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_lshlrev_b32_e32 v0, 4, v0
; GCN-NEXT:    v_lshrrev_b32_e64 v2, 6, s32
; GCN-NEXT:    v_add_u32_e32 v0, vcc, v0, v2
; GCN-NEXT:    v_mov_b32_e32 v2, 1
; GCN-NEXT:    v_mov_b32_e32 v1, 4
; GCN-NEXT:    buffer_store_dword v2, v0, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v2, 12, v0
; GCN-NEXT:    buffer_store_dword v1, v2, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v1, 8, v0
; GCN-NEXT:    v_mov_b32_e32 v2, 3
; GCN-NEXT:    buffer_store_dword v2, v1, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v0, 4, v0
; GCN-NEXT:    v_mov_b32_e32 v1, 2
; GCN-NEXT:    buffer_store_dword v1, v0, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN: ; ScratchSize: 144
  %alloca.align16 = alloca [8 x <4 x i32>], align 16, addrspace(5)
  %gep0 = getelementptr inbounds [8 x <4 x i32>], ptr addrspace(5) %alloca.align16, i32 0, i32 %idx
  store volatile <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr addrspace(5) %gep0, align 16
  ret void
}

define void @needs_align16_stack_align4(i32 %idx) #2 {
; GCN-LABEL: needs_align16_stack_align4:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0x3c0
; GCN-NEXT:    s_and_b32 s33, s33, 0xfffffc00
; GCN-NEXT:    v_lshlrev_b32_e32 v0, 4, v0
; GCN-NEXT:    v_lshrrev_b32_e64 v2, 6, s33
; GCN-NEXT:    v_add_u32_e32 v0, vcc, v0, v2
; GCN-NEXT:    v_mov_b32_e32 v2, 1
; GCN-NEXT:    v_mov_b32_e32 v1, 4
; GCN-NEXT:    buffer_store_dword v2, v0, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v2, 12, v0
; GCN-NEXT:    buffer_store_dword v1, v2, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v1, 8, v0
; GCN-NEXT:    v_mov_b32_e32 v2, 3
; GCN-NEXT:    s_mov_b32 s5, s34
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    s_addk_i32 s32, 0x2800
; GCN-NEXT:    buffer_store_dword v2, v1, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v0, 4, v0
; GCN-NEXT:    v_mov_b32_e32 v1, 2
; GCN-NEXT:    buffer_store_dword v1, v0, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    s_mov_b32 s34, s5
; GCN-NEXT:    s_mov_b32 s33, s4
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN: ; ScratchSize: 160
  %alloca.align16 = alloca [8 x <4 x i32>], align 16, addrspace(5)
  %gep0 = getelementptr inbounds [8 x <4 x i32>], ptr addrspace(5) %alloca.align16, i32 0, i32 %idx
  store volatile <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr addrspace(5) %gep0, align 16
  ret void
}


define void @needs_align32(i32 %idx) #0 {
; GCN-LABEL: needs_align32:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0x7c0
; GCN-NEXT:    s_and_b32 s33, s33, 0xfffff800
; GCN-NEXT:    v_lshlrev_b32_e32 v0, 4, v0
; GCN-NEXT:    v_lshrrev_b32_e64 v2, 6, s33
; GCN-NEXT:    v_add_u32_e32 v0, vcc, v0, v2
; GCN-NEXT:    v_mov_b32_e32 v2, 1
; GCN-NEXT:    v_mov_b32_e32 v1, 4
; GCN-NEXT:    buffer_store_dword v2, v0, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v2, 12, v0
; GCN-NEXT:    buffer_store_dword v1, v2, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v1, 8, v0
; GCN-NEXT:    v_mov_b32_e32 v2, 3
; GCN-NEXT:    s_mov_b32 s5, s34
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    s_addk_i32 s32, 0x3000
; GCN-NEXT:    buffer_store_dword v2, v1, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_or_b32_e32 v0, 4, v0
; GCN-NEXT:    v_mov_b32_e32 v1, 2
; GCN-NEXT:    buffer_store_dword v1, v0, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    s_mov_b32 s34, s5
; GCN-NEXT:    s_mov_b32 s33, s4
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN: ; ScratchSize: 192
  %alloca.align16 = alloca [8 x <4 x i32>], align 32, addrspace(5)
  %gep0 = getelementptr inbounds [8 x <4 x i32>], ptr addrspace(5) %alloca.align16, i32 0, i32 %idx
  store volatile <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr addrspace(5) %gep0, align 32
  ret void
}

define void @force_realign4(i32 %idx) #1 {
; GCN-LABEL: force_realign4:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0xc0
; GCN-NEXT:    s_and_b32 s33, s33, 0xffffff00
; GCN-NEXT:    v_lshlrev_b32_e32 v0, 2, v0
; GCN-NEXT:    v_lshrrev_b32_e64 v1, 6, s33
; GCN-NEXT:    s_mov_b32 s5, s34
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    s_addk_i32 s32, 0xd00
; GCN-NEXT:    v_add_u32_e32 v0, vcc, v0, v1
; GCN-NEXT:    v_mov_b32_e32 v1, 3
; GCN-NEXT:    buffer_store_dword v1, v0, s[0:3], 0 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    s_mov_b32 s34, s5
; GCN-NEXT:    s_mov_b32 s33, s4
; GCN-NEXT:    s_setpc_b64 s[30:31]
; GCN: ; ScratchSize: 52
  %alloca.align16 = alloca [8 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [8 x i32], ptr addrspace(5) %alloca.align16, i32 0, i32 %idx
  store volatile i32 3, ptr addrspace(5) %gep0, align 4
  ret void
}

define amdgpu_kernel void @kernel_call_align16_from_8() #0 {
; GCN-LABEL: kernel_call_align16_from_8:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_add_i32 s12, s12, s17
; GCN-NEXT:    s_lshr_b32 flat_scratch_hi, s12, 8
; GCN-NEXT:    s_add_u32 s0, s0, s17
; GCN-NEXT:    s_addc_u32 s1, s1, 0
; GCN-NEXT:    s_mov_b32 flat_scratch_lo, s13
; GCN-NEXT:    s_mov_b32 s13, s15
; GCN-NEXT:    s_mov_b32 s12, s14
; GCN-NEXT:    s_getpc_b64 s[14:15]
; GCN-NEXT:    s_add_u32 s14, s14, needs_align16_default_stack_align@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s15, s15, needs_align16_default_stack_align@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[18:19], s[14:15], 0x0
; GCN-NEXT:    v_lshlrev_b32_e32 v1, 10, v1
; GCN-NEXT:    v_lshlrev_b32_e32 v2, 20, v2
; GCN-NEXT:    v_or_b32_e32 v0, v0, v1
; GCN-NEXT:    v_mov_b32_e32 v3, 2
; GCN-NEXT:    v_or_b32_e32 v31, v0, v2
; GCN-NEXT:    s_mov_b32 s14, s16
; GCN-NEXT:    v_mov_b32_e32 v0, 1
; GCN-NEXT:    s_movk_i32 s32, 0x400
; GCN-NEXT:    buffer_store_dword v3, off, s[0:3], 0
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[18:19]
; GCN-NEXT:    s_endpgm
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 2, ptr addrspace(5) %alloca
  call void @needs_align16_default_stack_align(i32 1)
  ret void
}

; The call sequence should keep the stack on call aligned to 4
define amdgpu_kernel void @kernel_call_align16_from_5() {
; GCN-LABEL: kernel_call_align16_from_5:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_add_i32 s12, s12, s17
; GCN-NEXT:    s_lshr_b32 flat_scratch_hi, s12, 8
; GCN-NEXT:    s_add_u32 s0, s0, s17
; GCN-NEXT:    s_addc_u32 s1, s1, 0
; GCN-NEXT:    s_mov_b32 flat_scratch_lo, s13
; GCN-NEXT:    s_mov_b32 s13, s15
; GCN-NEXT:    s_mov_b32 s12, s14
; GCN-NEXT:    s_getpc_b64 s[14:15]
; GCN-NEXT:    s_add_u32 s14, s14, needs_align16_default_stack_align@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s15, s15, needs_align16_default_stack_align@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[18:19], s[14:15], 0x0
; GCN-NEXT:    v_lshlrev_b32_e32 v1, 10, v1
; GCN-NEXT:    v_lshlrev_b32_e32 v2, 20, v2
; GCN-NEXT:    v_or_b32_e32 v0, v0, v1
; GCN-NEXT:    v_mov_b32_e32 v3, 2
; GCN-NEXT:    v_or_b32_e32 v31, v0, v2
; GCN-NEXT:    s_mov_b32 s14, s16
; GCN-NEXT:    v_mov_b32_e32 v0, 1
; GCN-NEXT:    s_movk_i32 s32, 0x400
; GCN-NEXT:    buffer_store_byte v3, off, s[0:3], 0
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[18:19]
; GCN-NEXT:    s_endpgm
  %alloca0 = alloca i8, align 1, addrspace(5)
  store volatile i8 2, ptr  addrspace(5) %alloca0

  call void @needs_align16_default_stack_align(i32 1)
  ret void
}

define amdgpu_kernel void @kernel_call_align4_from_5() {
; GCN-LABEL: kernel_call_align4_from_5:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_add_i32 s12, s12, s17
; GCN-NEXT:    s_lshr_b32 flat_scratch_hi, s12, 8
; GCN-NEXT:    s_add_u32 s0, s0, s17
; GCN-NEXT:    s_addc_u32 s1, s1, 0
; GCN-NEXT:    s_mov_b32 flat_scratch_lo, s13
; GCN-NEXT:    s_mov_b32 s13, s15
; GCN-NEXT:    s_mov_b32 s12, s14
; GCN-NEXT:    s_getpc_b64 s[14:15]
; GCN-NEXT:    s_add_u32 s14, s14, needs_align16_stack_align4@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s15, s15, needs_align16_stack_align4@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[18:19], s[14:15], 0x0
; GCN-NEXT:    v_lshlrev_b32_e32 v1, 10, v1
; GCN-NEXT:    v_lshlrev_b32_e32 v2, 20, v2
; GCN-NEXT:    v_or_b32_e32 v0, v0, v1
; GCN-NEXT:    v_mov_b32_e32 v3, 2
; GCN-NEXT:    v_or_b32_e32 v31, v0, v2
; GCN-NEXT:    s_mov_b32 s14, s16
; GCN-NEXT:    v_mov_b32_e32 v0, 1
; GCN-NEXT:    s_movk_i32 s32, 0x400
; GCN-NEXT:    buffer_store_byte v3, off, s[0:3], 0
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[18:19]
; GCN-NEXT:    s_endpgm
  %alloca0 = alloca i8, align 1, addrspace(5)
  store volatile i8 2, ptr  addrspace(5) %alloca0

  call void @needs_align16_stack_align4(i32 1)
  ret void
}

define void @default_realign_align128(i32 %idx) #0 {
; GCN-LABEL: default_realign_align128:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0x1fc0
; GCN-NEXT:    s_and_b32 s33, s33, 0xffffe000
; GCN-NEXT:    s_mov_b32 s5, s34
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    s_addk_i32 s32, 0x4000
; GCN-NEXT:    v_mov_b32_e32 v0, 9
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s33
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    s_mov_b32 s34, s5
; GCN-NEXT:    s_mov_b32 s33, s4
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %alloca.align = alloca i32, align 128, addrspace(5)
  store volatile i32 9, ptr addrspace(5) %alloca.align, align 128
  ret void
}

define void @disable_realign_align128(i32 %idx) #3 {
; GCN-LABEL: disable_realign_align128:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_mov_b32_e32 v0, 9
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s32
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %alloca.align = alloca i32, align 128, addrspace(5)
  store volatile i32 9, ptr addrspace(5) %alloca.align, align 128
  ret void
}

declare void @extern_func(<32 x i32>, i32) #0
define void @func_call_align1024_bp_gets_vgpr_spill(<32 x i32> %a, i32 %b) #0 {
; The test forces the stack to be realigned to a new boundary
; since there is a local object with an alignment of 1024.
; Should use BP to access the incoming stack arguments.
; The BP value is saved/restored with a VGPR spill.
; GCN-LABEL: func_call_align1024_bp_gets_vgpr_spill:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s16, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0xffc0
; GCN-NEXT:    s_and_b32 s33, s33, 0xffff0000
; GCN-NEXT:    s_or_saveexec_b64 s[18:19], -1
; GCN-NEXT:    buffer_store_dword v40, off, s[0:3], s33 offset:1028 ; 4-byte Folded Spill
; GCN-NEXT:    s_mov_b64 exec, s[18:19]
; GCN-NEXT:    v_writelane_b32 v40, s16, 2
; GCN-NEXT:    v_mov_b32_e32 v32, 0
; GCN-NEXT:    v_writelane_b32 v40, s34, 3
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    buffer_store_dword v32, off, s[0:3], s33 offset:1024
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    buffer_load_dword v32, off, s[0:3], s34
; GCN-NEXT:    buffer_load_dword v33, off, s[0:3], s34 offset:4
; GCN-NEXT:    s_add_i32 s32, s32, 0x30000
; GCN-NEXT:    s_getpc_b64 s[16:17]
; GCN-NEXT:    s_add_u32 s16, s16, extern_func@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s17, s17, extern_func@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s[16:17], s[16:17], 0x0
; GCN-NEXT:    v_writelane_b32 v40, s30, 0
; GCN-NEXT:    v_writelane_b32 v40, s31, 1
; GCN-NEXT:    s_waitcnt vmcnt(1)
; GCN-NEXT:    buffer_store_dword v32, off, s[0:3], s32
; GCN-NEXT:    s_waitcnt vmcnt(1)
; GCN-NEXT:    buffer_store_dword v33, off, s[0:3], s32 offset:4
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_swappc_b64 s[30:31], s[16:17]
; GCN-NEXT:    v_readlane_b32 s31, v40, 1
; GCN-NEXT:    v_readlane_b32 s30, v40, 0
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    v_readlane_b32 s4, v40, 2
; GCN-NEXT:    v_readlane_b32 s34, v40, 3
; GCN-NEXT:    s_or_saveexec_b64 s[6:7], -1
; GCN-NEXT:    buffer_load_dword v40, off, s[0:3], s33 offset:1028 ; 4-byte Folded Reload
; GCN-NEXT:    s_mov_b64 exec, s[6:7]
; GCN-NEXT:    s_mov_b32 s33, s4
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]

  %temp = alloca i32, align 1024, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 1024
  call void @extern_func(<32 x i32> %a, i32 %b)
  ret void
}

%struct.Data = type { [9 x i32] }
define i32 @needs_align1024_stack_args_used_inside_loop(ptr addrspace(5) nocapture readonly byval(%struct.Data) align 8 %arg) local_unnamed_addr #4 {
; The local object allocation needed an alignment of 1024.
; Since the function argument is accessed in a loop with an
; index variable, the base pointer first get loaded into a VGPR
; and that value should be further referenced to load the incoming values.
; The BP value will get saved/restored in an SGPR at the prolgoue/epilogue.
; GCN-LABEL: needs_align1024_stack_args_used_inside_loop:
; GCN:       ; %bb.0: ; %begin
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s11, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0xffc0
; GCN-NEXT:    s_mov_b32 s14, s34
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    s_and_b32 s33, s33, 0xffff0000
; GCN-NEXT:    v_lshrrev_b32_e64 v1, 6, s34
; GCN-NEXT:    v_mov_b32_e32 v0, 0
; GCN-NEXT:    s_mov_b32 s10, 0
; GCN-NEXT:    s_mov_b64 s[4:5], 0
; GCN-NEXT:    s_add_i32 s32, s32, 0x30000
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:1024
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    ; implicit-def: $sgpr6_sgpr7
; GCN-NEXT:    s_branch .LBB10_2
; GCN-NEXT:  .LBB10_1: ; %Flow
; GCN-NEXT:    ; in Loop: Header=BB10_2 Depth=1
; GCN-NEXT:    s_or_b64 exec, exec, s[8:9]
; GCN-NEXT:    s_and_b64 s[8:9], exec, s[6:7]
; GCN-NEXT:    s_or_b64 s[4:5], s[8:9], s[4:5]
; GCN-NEXT:    s_andn2_b64 exec, exec, s[4:5]
; GCN-NEXT:    s_cbranch_execz .LBB10_4
; GCN-NEXT:  .LBB10_2: ; %loop_body
; GCN-NEXT:    ; =>This Inner Loop Header: Depth=1
; GCN-NEXT:    buffer_load_dword v0, v1, s[0:3], 0 offen
; GCN-NEXT:    s_or_b64 s[6:7], s[6:7], exec
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_cmp_eq_u32_e32 vcc, s10, v0
; GCN-NEXT:    v_mov_b32_e32 v0, 0
; GCN-NEXT:    s_and_saveexec_b64 s[8:9], vcc
; GCN-NEXT:    s_cbranch_execz .LBB10_1
; GCN-NEXT:  ; %bb.3: ; %loop_end
; GCN-NEXT:    ; in Loop: Header=BB10_2 Depth=1
; GCN-NEXT:    s_add_i32 s10, s10, 1
; GCN-NEXT:    s_cmp_eq_u32 s10, 9
; GCN-NEXT:    s_cselect_b64 s[12:13], -1, 0
; GCN-NEXT:    s_andn2_b64 s[6:7], s[6:7], exec
; GCN-NEXT:    s_and_b64 s[12:13], s[12:13], exec
; GCN-NEXT:    v_add_u32_e32 v1, vcc, 4, v1
; GCN-NEXT:    v_mov_b32_e32 v0, 1
; GCN-NEXT:    s_or_b64 s[6:7], s[6:7], s[12:13]
; GCN-NEXT:    s_branch .LBB10_1
; GCN-NEXT:  .LBB10_4: ; %exit
; GCN-NEXT:    s_or_b64 exec, exec, s[4:5]
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    s_mov_b32 s34, s14
; GCN-NEXT:    s_mov_b32 s33, s11
; GCN-NEXT:    s_setpc_b64 s[30:31]
begin:
  %local_var = alloca i32, align 1024, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %local_var, align 1024
  br label %loop_body

loop_end:                                                ; preds = %loop_body
  %idx_next = add nuw nsw i32 %lp_idx, 1
  %lp_exit_cond = icmp eq i32 %idx_next, 9
  br i1 %lp_exit_cond, label %exit, label %loop_body

loop_body:                                                ; preds = %loop_end, %begin
  %lp_idx = phi i32 [ 0, %begin ], [ %idx_next, %loop_end ]
  %ptr = getelementptr inbounds %struct.Data, ptr addrspace(5) %arg, i32 0, i32 0, i32 %lp_idx
  %val = load i32, ptr addrspace(5) %ptr, align 8
  %lp_cond = icmp eq i32 %val, %lp_idx
  br i1 %lp_cond, label %loop_end, label %exit

exit:                                               ; preds = %loop_end, %loop_body
  %out = phi i32 [ 0, %loop_body ], [ 1, %loop_end ]
  ret i32 %out
}

define void @no_free_scratch_sgpr_for_bp_copy(<32 x i32> %a, i32 %b) #0 {
; GCN-LABEL: no_free_scratch_sgpr_for_bp_copy:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 vcc_lo, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0x1fc0
; GCN-NEXT:    s_and_b32 s33, s33, 0xffffe000
; GCN-NEXT:    s_xor_saveexec_b64 s[4:5], -1
; GCN-NEXT:    buffer_store_dword v1, off, s[0:3], s33 offset:132 ; 4-byte Folded Spill
; GCN-NEXT:    s_mov_b64 exec, s[4:5]
; GCN-NEXT:    v_writelane_b32 v1, s34, 0
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s34 offset:4
; GCN-NEXT:    s_addk_i32 s32, 0x6000
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    v_readlane_b32 s34, v1, 0
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:128
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:    s_xor_saveexec_b64 s[4:5], -1
; GCN-NEXT:    buffer_load_dword v1, off, s[0:3], s33 offset:132 ; 4-byte Folded Reload
; GCN-NEXT:    s_mov_b64 exec, s[4:5]
; GCN-NEXT:    s_mov_b32 s33, vcc_lo
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %local_val = alloca i32, align 128, addrspace(5)
  store volatile i32 %b, ptr addrspace(5) %local_val, align 128
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
; GCN-LABEL: no_free_regs_spill_bp_to_memory:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0x1fc0
; GCN-NEXT:    s_and_b32 s33, s33, 0xffffe000
; GCN-NEXT:    s_xor_saveexec_b64 s[6:7], -1
; GCN-NEXT:    buffer_store_dword v39, off, s[0:3], s33 offset:132 ; 4-byte Folded Spill
; GCN-NEXT:    s_mov_b64 exec, s[6:7]
; GCN-NEXT:    v_mov_b32_e32 v0, s4
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:136 ; 4-byte Folded Spill
; GCN-NEXT:    v_mov_b32_e32 v0, s34
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:140 ; 4-byte Folded Spill
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s34 offset:4
; GCN-NEXT:    v_writelane_b32 v39, s39, 0
; GCN-NEXT:    v_writelane_b32 v39, s40, 1
; GCN-NEXT:    v_writelane_b32 v39, s41, 2
; GCN-NEXT:    v_writelane_b32 v39, s42, 3
; GCN-NEXT:    v_writelane_b32 v39, s43, 4
; GCN-NEXT:    v_writelane_b32 v39, s44, 5
; GCN-NEXT:    v_writelane_b32 v39, s45, 6
; GCN-NEXT:    v_writelane_b32 v39, s46, 7
; GCN-NEXT:    v_writelane_b32 v39, s47, 8
; GCN-NEXT:    v_writelane_b32 v39, s48, 9
; GCN-NEXT:    v_writelane_b32 v39, s49, 10
; GCN-NEXT:    v_writelane_b32 v39, s50, 11
; GCN-NEXT:    v_writelane_b32 v39, s51, 12
; GCN-NEXT:    v_writelane_b32 v39, s52, 13
; GCN-NEXT:    v_writelane_b32 v39, s53, 14
; GCN-NEXT:    v_writelane_b32 v39, s54, 15
; GCN-NEXT:    v_writelane_b32 v39, s55, 16
; GCN-NEXT:    v_writelane_b32 v39, s56, 17
; GCN-NEXT:    v_writelane_b32 v39, s57, 18
; GCN-NEXT:    v_writelane_b32 v39, s58, 19
; GCN-NEXT:    v_writelane_b32 v39, s59, 20
; GCN-NEXT:    v_writelane_b32 v39, s60, 21
; GCN-NEXT:    v_writelane_b32 v39, s61, 22
; GCN-NEXT:    v_writelane_b32 v39, s62, 23
; GCN-NEXT:    v_writelane_b32 v39, s63, 24
; GCN-NEXT:    v_writelane_b32 v39, s64, 25
; GCN-NEXT:    v_writelane_b32 v39, s65, 26
; GCN-NEXT:    v_writelane_b32 v39, s66, 27
; GCN-NEXT:    v_writelane_b32 v39, s67, 28
; GCN-NEXT:    v_writelane_b32 v39, s68, 29
; GCN-NEXT:    v_writelane_b32 v39, s69, 30
; GCN-NEXT:    v_writelane_b32 v39, s70, 31
; GCN-NEXT:    v_writelane_b32 v39, s71, 32
; GCN-NEXT:    v_writelane_b32 v39, s72, 33
; GCN-NEXT:    v_writelane_b32 v39, s73, 34
; GCN-NEXT:    v_writelane_b32 v39, s74, 35
; GCN-NEXT:    v_writelane_b32 v39, s75, 36
; GCN-NEXT:    v_writelane_b32 v39, s76, 37
; GCN-NEXT:    v_writelane_b32 v39, s77, 38
; GCN-NEXT:    v_writelane_b32 v39, s78, 39
; GCN-NEXT:    v_writelane_b32 v39, s79, 40
; GCN-NEXT:    v_writelane_b32 v39, s80, 41
; GCN-NEXT:    v_writelane_b32 v39, s81, 42
; GCN-NEXT:    v_writelane_b32 v39, s82, 43
; GCN-NEXT:    v_writelane_b32 v39, s83, 44
; GCN-NEXT:    v_writelane_b32 v39, s84, 45
; GCN-NEXT:    v_writelane_b32 v39, s85, 46
; GCN-NEXT:    v_writelane_b32 v39, s86, 47
; GCN-NEXT:    v_writelane_b32 v39, s87, 48
; GCN-NEXT:    v_writelane_b32 v39, s88, 49
; GCN-NEXT:    v_writelane_b32 v39, s89, 50
; GCN-NEXT:    v_writelane_b32 v39, s90, 51
; GCN-NEXT:    v_writelane_b32 v39, s91, 52
; GCN-NEXT:    v_writelane_b32 v39, s92, 53
; GCN-NEXT:    v_writelane_b32 v39, s93, 54
; GCN-NEXT:    v_writelane_b32 v39, s94, 55
; GCN-NEXT:    v_writelane_b32 v39, s95, 56
; GCN-NEXT:    v_writelane_b32 v39, s96, 57
; GCN-NEXT:    v_writelane_b32 v39, s97, 58
; GCN-NEXT:    v_writelane_b32 v39, s98, 59
; GCN-NEXT:    v_writelane_b32 v39, s99, 60
; GCN-NEXT:    v_writelane_b32 v39, s100, 61
; GCN-NEXT:    v_writelane_b32 v39, s101, 62
; GCN-NEXT:    v_writelane_b32 v39, s102, 63
; GCN-NEXT:    s_addk_i32 s32, 0x6000
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:128
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:    ; clobber nonpreserved SGPRs and 64 CSRs
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:    ; clobber all VGPRs
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:136 ; 4-byte Folded Reload
; GCN-NEXT:    v_readlane_b32 s102, v39, 63
; GCN-NEXT:    v_readlane_b32 s101, v39, 62
; GCN-NEXT:    v_readlane_b32 s100, v39, 61
; GCN-NEXT:    v_readlane_b32 s99, v39, 60
; GCN-NEXT:    v_readlane_b32 s98, v39, 59
; GCN-NEXT:    v_readlane_b32 s97, v39, 58
; GCN-NEXT:    v_readlane_b32 s96, v39, 57
; GCN-NEXT:    v_readlane_b32 s95, v39, 56
; GCN-NEXT:    v_readlane_b32 s94, v39, 55
; GCN-NEXT:    v_readlane_b32 s93, v39, 54
; GCN-NEXT:    v_readlane_b32 s92, v39, 53
; GCN-NEXT:    v_readlane_b32 s91, v39, 52
; GCN-NEXT:    v_readlane_b32 s90, v39, 51
; GCN-NEXT:    v_readlane_b32 s89, v39, 50
; GCN-NEXT:    v_readlane_b32 s88, v39, 49
; GCN-NEXT:    v_readlane_b32 s87, v39, 48
; GCN-NEXT:    v_readlane_b32 s86, v39, 47
; GCN-NEXT:    v_readlane_b32 s85, v39, 46
; GCN-NEXT:    v_readlane_b32 s84, v39, 45
; GCN-NEXT:    v_readlane_b32 s83, v39, 44
; GCN-NEXT:    v_readlane_b32 s82, v39, 43
; GCN-NEXT:    v_readlane_b32 s81, v39, 42
; GCN-NEXT:    v_readlane_b32 s80, v39, 41
; GCN-NEXT:    v_readlane_b32 s79, v39, 40
; GCN-NEXT:    v_readlane_b32 s78, v39, 39
; GCN-NEXT:    v_readlane_b32 s77, v39, 38
; GCN-NEXT:    v_readlane_b32 s76, v39, 37
; GCN-NEXT:    v_readlane_b32 s75, v39, 36
; GCN-NEXT:    v_readlane_b32 s74, v39, 35
; GCN-NEXT:    v_readlane_b32 s73, v39, 34
; GCN-NEXT:    v_readlane_b32 s72, v39, 33
; GCN-NEXT:    v_readlane_b32 s71, v39, 32
; GCN-NEXT:    v_readlane_b32 s70, v39, 31
; GCN-NEXT:    v_readlane_b32 s69, v39, 30
; GCN-NEXT:    v_readlane_b32 s68, v39, 29
; GCN-NEXT:    v_readlane_b32 s67, v39, 28
; GCN-NEXT:    v_readlane_b32 s66, v39, 27
; GCN-NEXT:    v_readlane_b32 s65, v39, 26
; GCN-NEXT:    v_readlane_b32 s64, v39, 25
; GCN-NEXT:    v_readlane_b32 s63, v39, 24
; GCN-NEXT:    v_readlane_b32 s62, v39, 23
; GCN-NEXT:    v_readlane_b32 s61, v39, 22
; GCN-NEXT:    v_readlane_b32 s60, v39, 21
; GCN-NEXT:    v_readlane_b32 s59, v39, 20
; GCN-NEXT:    v_readlane_b32 s58, v39, 19
; GCN-NEXT:    v_readlane_b32 s57, v39, 18
; GCN-NEXT:    v_readlane_b32 s56, v39, 17
; GCN-NEXT:    v_readlane_b32 s55, v39, 16
; GCN-NEXT:    v_readlane_b32 s54, v39, 15
; GCN-NEXT:    v_readlane_b32 s53, v39, 14
; GCN-NEXT:    v_readlane_b32 s52, v39, 13
; GCN-NEXT:    v_readlane_b32 s51, v39, 12
; GCN-NEXT:    v_readlane_b32 s50, v39, 11
; GCN-NEXT:    v_readlane_b32 s49, v39, 10
; GCN-NEXT:    v_readlane_b32 s48, v39, 9
; GCN-NEXT:    v_readlane_b32 s47, v39, 8
; GCN-NEXT:    v_readlane_b32 s46, v39, 7
; GCN-NEXT:    v_readlane_b32 s45, v39, 6
; GCN-NEXT:    v_readlane_b32 s44, v39, 5
; GCN-NEXT:    v_readlane_b32 s43, v39, 4
; GCN-NEXT:    v_readlane_b32 s42, v39, 3
; GCN-NEXT:    v_readlane_b32 s41, v39, 2
; GCN-NEXT:    v_readlane_b32 s40, v39, 1
; GCN-NEXT:    v_readlane_b32 s39, v39, 0
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_readfirstlane_b32 s4, v0
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:140 ; 4-byte Folded Reload
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_readfirstlane_b32 s34, v0
; GCN-NEXT:    s_xor_saveexec_b64 s[6:7], -1
; GCN-NEXT:    buffer_load_dword v39, off, s[0:3], s33 offset:132 ; 4-byte Folded Reload
; GCN-NEXT:    s_mov_b64 exec, s[6:7]
; GCN-NEXT:    s_mov_b32 s33, s4
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %local_val = alloca i32, align 128, addrspace(5)
  store volatile i32 %b, ptr addrspace(5) %local_val, align 128

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

define void @spill_bp_to_memory_scratch_reg_needed_mubuf_offset(<32 x i32> %a, i32 %b, ptr addrspace(5) byval([4096 x i8]) align 4 %arg) #5 {
; If the size of the offset exceeds the MUBUF offset field we need another
; scratch VGPR to hold the offset.
; GCN-LABEL: spill_bp_to_memory_scratch_reg_needed_mubuf_offset:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 s4, s33
; GCN-NEXT:    s_add_i32 s33, s32, 0x1fc0
; GCN-NEXT:    s_and_b32 s33, s33, 0xffffe000
; GCN-NEXT:    s_xor_saveexec_b64 s[6:7], -1
; GCN-NEXT:    s_add_i32 s5, s33, 0x42100
; GCN-NEXT:    buffer_store_dword v39, off, s[0:3], s5 ; 4-byte Folded Spill
; GCN-NEXT:    s_mov_b64 exec, s[6:7]
; GCN-NEXT:    v_mov_b32_e32 v0, s4
; GCN-NEXT:    s_add_i32 s5, s33, 0x42200
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s5 ; 4-byte Folded Spill
; GCN-NEXT:    v_mov_b32_e32 v0, s34
; GCN-NEXT:    s_add_i32 s5, s33, 0x42300
; GCN-NEXT:    s_mov_b32 s34, s32
; GCN-NEXT:    buffer_store_dword v0, off, s[0:3], s5 ; 4-byte Folded Spill
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s34 offset:4
; GCN-NEXT:    v_writelane_b32 v39, s39, 0
; GCN-NEXT:    v_writelane_b32 v39, s40, 1
; GCN-NEXT:    v_writelane_b32 v39, s41, 2
; GCN-NEXT:    v_writelane_b32 v39, s42, 3
; GCN-NEXT:    v_writelane_b32 v39, s43, 4
; GCN-NEXT:    v_writelane_b32 v39, s44, 5
; GCN-NEXT:    v_writelane_b32 v39, s45, 6
; GCN-NEXT:    v_writelane_b32 v39, s46, 7
; GCN-NEXT:    v_writelane_b32 v39, s47, 8
; GCN-NEXT:    v_writelane_b32 v39, s48, 9
; GCN-NEXT:    v_writelane_b32 v39, s49, 10
; GCN-NEXT:    v_writelane_b32 v39, s50, 11
; GCN-NEXT:    v_writelane_b32 v39, s51, 12
; GCN-NEXT:    v_writelane_b32 v39, s52, 13
; GCN-NEXT:    v_writelane_b32 v39, s53, 14
; GCN-NEXT:    v_writelane_b32 v39, s54, 15
; GCN-NEXT:    v_writelane_b32 v39, s55, 16
; GCN-NEXT:    v_writelane_b32 v39, s56, 17
; GCN-NEXT:    v_writelane_b32 v39, s57, 18
; GCN-NEXT:    v_writelane_b32 v39, s58, 19
; GCN-NEXT:    v_writelane_b32 v39, s59, 20
; GCN-NEXT:    v_writelane_b32 v39, s60, 21
; GCN-NEXT:    v_writelane_b32 v39, s61, 22
; GCN-NEXT:    v_writelane_b32 v39, s62, 23
; GCN-NEXT:    v_writelane_b32 v39, s63, 24
; GCN-NEXT:    v_writelane_b32 v39, s64, 25
; GCN-NEXT:    v_writelane_b32 v39, s65, 26
; GCN-NEXT:    v_writelane_b32 v39, s66, 27
; GCN-NEXT:    v_writelane_b32 v39, s67, 28
; GCN-NEXT:    v_writelane_b32 v39, s68, 29
; GCN-NEXT:    v_writelane_b32 v39, s69, 30
; GCN-NEXT:    v_writelane_b32 v39, s70, 31
; GCN-NEXT:    v_writelane_b32 v39, s71, 32
; GCN-NEXT:    v_writelane_b32 v39, s72, 33
; GCN-NEXT:    v_writelane_b32 v39, s73, 34
; GCN-NEXT:    v_writelane_b32 v39, s74, 35
; GCN-NEXT:    v_writelane_b32 v39, s75, 36
; GCN-NEXT:    v_writelane_b32 v39, s76, 37
; GCN-NEXT:    v_writelane_b32 v39, s77, 38
; GCN-NEXT:    v_writelane_b32 v39, s78, 39
; GCN-NEXT:    v_writelane_b32 v39, s79, 40
; GCN-NEXT:    v_writelane_b32 v39, s80, 41
; GCN-NEXT:    v_writelane_b32 v39, s81, 42
; GCN-NEXT:    v_writelane_b32 v39, s82, 43
; GCN-NEXT:    v_writelane_b32 v39, s83, 44
; GCN-NEXT:    v_writelane_b32 v39, s84, 45
; GCN-NEXT:    v_writelane_b32 v39, s85, 46
; GCN-NEXT:    v_writelane_b32 v39, s86, 47
; GCN-NEXT:    v_writelane_b32 v39, s87, 48
; GCN-NEXT:    v_writelane_b32 v39, s88, 49
; GCN-NEXT:    v_writelane_b32 v39, s89, 50
; GCN-NEXT:    v_writelane_b32 v39, s90, 51
; GCN-NEXT:    v_writelane_b32 v39, s91, 52
; GCN-NEXT:    v_writelane_b32 v39, s92, 53
; GCN-NEXT:    v_writelane_b32 v39, s93, 54
; GCN-NEXT:    v_writelane_b32 v39, s94, 55
; GCN-NEXT:    v_writelane_b32 v39, s95, 56
; GCN-NEXT:    v_writelane_b32 v39, s96, 57
; GCN-NEXT:    v_writelane_b32 v39, s97, 58
; GCN-NEXT:    v_writelane_b32 v39, s98, 59
; GCN-NEXT:    v_writelane_b32 v39, s99, 60
; GCN-NEXT:    v_writelane_b32 v39, s100, 61
; GCN-NEXT:    v_writelane_b32 v39, s101, 62
; GCN-NEXT:    v_mov_b32_e32 v1, 0x1080
; GCN-NEXT:    v_writelane_b32 v39, s102, 63
; GCN-NEXT:    s_add_i32 s32, s32, 0x46000
; GCN-NEXT:    s_mov_b32 s32, s34
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    buffer_store_dword v0, v1, s[0:3], s33 offen
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:    ; clobber nonpreserved SGPRs and 64 CSRs
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:    ;;#ASMSTART
; GCN-NEXT:    ; clobber all VGPRs
; GCN-NEXT:    ;;#ASMEND
; GCN-NEXT:    s_add_i32 s5, s33, 0x42200
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s5 ; 4-byte Folded Reload
; GCN-NEXT:    s_add_i32 s5, s33, 0x42300
; GCN-NEXT:    v_readlane_b32 s102, v39, 63
; GCN-NEXT:    v_readlane_b32 s101, v39, 62
; GCN-NEXT:    v_readlane_b32 s100, v39, 61
; GCN-NEXT:    v_readlane_b32 s99, v39, 60
; GCN-NEXT:    v_readlane_b32 s98, v39, 59
; GCN-NEXT:    v_readlane_b32 s97, v39, 58
; GCN-NEXT:    v_readlane_b32 s96, v39, 57
; GCN-NEXT:    v_readlane_b32 s95, v39, 56
; GCN-NEXT:    v_readlane_b32 s94, v39, 55
; GCN-NEXT:    v_readlane_b32 s93, v39, 54
; GCN-NEXT:    v_readlane_b32 s92, v39, 53
; GCN-NEXT:    v_readlane_b32 s91, v39, 52
; GCN-NEXT:    v_readlane_b32 s90, v39, 51
; GCN-NEXT:    v_readlane_b32 s89, v39, 50
; GCN-NEXT:    v_readlane_b32 s88, v39, 49
; GCN-NEXT:    v_readlane_b32 s87, v39, 48
; GCN-NEXT:    v_readlane_b32 s86, v39, 47
; GCN-NEXT:    v_readlane_b32 s85, v39, 46
; GCN-NEXT:    v_readlane_b32 s84, v39, 45
; GCN-NEXT:    v_readlane_b32 s83, v39, 44
; GCN-NEXT:    v_readlane_b32 s82, v39, 43
; GCN-NEXT:    v_readlane_b32 s81, v39, 42
; GCN-NEXT:    v_readlane_b32 s80, v39, 41
; GCN-NEXT:    v_readlane_b32 s79, v39, 40
; GCN-NEXT:    v_readlane_b32 s78, v39, 39
; GCN-NEXT:    v_readlane_b32 s77, v39, 38
; GCN-NEXT:    v_readlane_b32 s76, v39, 37
; GCN-NEXT:    v_readlane_b32 s75, v39, 36
; GCN-NEXT:    v_readlane_b32 s74, v39, 35
; GCN-NEXT:    v_readlane_b32 s73, v39, 34
; GCN-NEXT:    v_readlane_b32 s72, v39, 33
; GCN-NEXT:    v_readlane_b32 s71, v39, 32
; GCN-NEXT:    v_readlane_b32 s70, v39, 31
; GCN-NEXT:    v_readlane_b32 s69, v39, 30
; GCN-NEXT:    v_readlane_b32 s68, v39, 29
; GCN-NEXT:    v_readlane_b32 s67, v39, 28
; GCN-NEXT:    v_readlane_b32 s66, v39, 27
; GCN-NEXT:    v_readlane_b32 s65, v39, 26
; GCN-NEXT:    v_readlane_b32 s64, v39, 25
; GCN-NEXT:    v_readlane_b32 s63, v39, 24
; GCN-NEXT:    v_readlane_b32 s62, v39, 23
; GCN-NEXT:    v_readlane_b32 s61, v39, 22
; GCN-NEXT:    v_readlane_b32 s60, v39, 21
; GCN-NEXT:    v_readlane_b32 s59, v39, 20
; GCN-NEXT:    v_readlane_b32 s58, v39, 19
; GCN-NEXT:    v_readlane_b32 s57, v39, 18
; GCN-NEXT:    v_readlane_b32 s56, v39, 17
; GCN-NEXT:    v_readlane_b32 s55, v39, 16
; GCN-NEXT:    v_readlane_b32 s54, v39, 15
; GCN-NEXT:    v_readlane_b32 s53, v39, 14
; GCN-NEXT:    v_readlane_b32 s52, v39, 13
; GCN-NEXT:    v_readlane_b32 s51, v39, 12
; GCN-NEXT:    v_readlane_b32 s50, v39, 11
; GCN-NEXT:    v_readlane_b32 s49, v39, 10
; GCN-NEXT:    v_readlane_b32 s48, v39, 9
; GCN-NEXT:    v_readlane_b32 s47, v39, 8
; GCN-NEXT:    v_readlane_b32 s46, v39, 7
; GCN-NEXT:    v_readlane_b32 s45, v39, 6
; GCN-NEXT:    v_readlane_b32 s44, v39, 5
; GCN-NEXT:    v_readlane_b32 s43, v39, 4
; GCN-NEXT:    v_readlane_b32 s42, v39, 3
; GCN-NEXT:    v_readlane_b32 s41, v39, 2
; GCN-NEXT:    v_readlane_b32 s40, v39, 1
; GCN-NEXT:    v_readlane_b32 s39, v39, 0
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_readfirstlane_b32 s4, v0
; GCN-NEXT:    buffer_load_dword v0, off, s[0:3], s5 ; 4-byte Folded Reload
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_readfirstlane_b32 s34, v0
; GCN-NEXT:    s_xor_saveexec_b64 s[6:7], -1
; GCN-NEXT:    s_add_i32 s5, s33, 0x42100
; GCN-NEXT:    buffer_load_dword v39, off, s[0:3], s5 ; 4-byte Folded Reload
; GCN-NEXT:    s_mov_b64 exec, s[6:7]
; GCN-NEXT:    s_mov_b32 s33, s4
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %local_val = alloca i32, align 128, addrspace(5)
  store volatile i32 %b, ptr addrspace(5) %local_val, align 128

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
