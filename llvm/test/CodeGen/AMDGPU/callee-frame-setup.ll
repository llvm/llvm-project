; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck  -enable-var-scope -check-prefix=GCN -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}callee_no_stack:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack() #0 {
  ret void
}

; GCN-LABEL: {{^}}callee_no_stack_no_fp_elim_all:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_mov_b32 s4, s33
; GCN-NEXT: s_mov_b32 s33, s32
; GCN-NEXT: s_mov_b32 s33, s4
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack_no_fp_elim_all() #1 {
  ret void
}

; GCN-LABEL: {{^}}callee_no_stack_no_fp_elim_nonleaf:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack_no_fp_elim_nonleaf() #2 {
  ret void
}

; GCN-LABEL: {{^}}callee_with_stack:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: v_mov_b32_e32 v0, 0{{$}}
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], s32{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; Can use free call clobbered register to preserve original FP value.

; GCN-LABEL: {{^}}callee_with_stack_no_fp_elim_all:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_mov_b32 s4, s33
; GCN-NEXT: s_mov_b32 s33, s32
; GCN-NEXT: s_add_u32 s32, s32, 0x200
; GCN-NEXT: v_mov_b32_e32 v0, 0{{$}}
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], s33 offset:4{{$}}
; GCN-NEXT: s_sub_u32 s32, s32, 0x200
; GCN-NEXT: s_mov_b32 s33, s4
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack_no_fp_elim_all() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}callee_with_stack_no_fp_elim_non_leaf:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN-NEXT: v_mov_b32_e32 v0, 0{{$}}
; GCN-NEXT: buffer_store_dword v0, off, s[0:3], s32{{$}}
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack_no_fp_elim_non_leaf() #2 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}callee_with_stack_and_call:
; GCN: ; %bb.0:
; GCN-NEXT: s_waitcnt
; GCN: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN: v_writelane_b32 [[CSR_VGPR]], s33, 2
; GCN-DAG: s_mov_b32 s33, s32
; GCN-DAG: s_add_u32 s32, s32, 0x400{{$}}
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s30,
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31,

; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s33{{$}}

; GCN: s_swappc_b64

; GCN-DAG: v_readlane_b32 s5, [[CSR_VGPR]]
; GCN-DAG: v_readlane_b32 s4, [[CSR_VGPR]]

; GCN: s_sub_u32 s32, s32, 0x400{{$}}
; GCN-NEXT: v_readlane_b32 s33, [[CSR_VGPR]], 2
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 offset:4 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)

; GCN-NEXT: s_setpc_b64
define void @callee_with_stack_and_call() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void @external_void_func_void()
  ret void
}

; Should be able to copy incoming stack pointer directly to inner
; call's stack pointer argument.

; There is stack usage only because of the need to evict a VGPR for
; spilling CSR SGPRs.

; GCN-LABEL: {{^}}callee_no_stack_with_call:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-DAG: s_add_u32 s32, s32, 0x400
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s33, [[FP_SPILL_LANE:[0-9]+]]

; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s30, 0
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31, 1
; GCN: s_swappc_b64

; GCN-DAG: v_readlane_b32 s4, v40, 0
; GCN-DAG: v_readlane_b32 s5, v40, 1

; GCN: s_sub_u32 s32, s32, 0x400
; GCN-NEXT: v_readlane_b32 s33, [[CSR_VGPR]], [[FP_SPILL_LANE]]
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @callee_no_stack_with_call() #0 {
  call void @external_void_func_void()
  ret void
}

declare hidden void @external_void_func_void() #0

; Make sure if a CSR vgpr is used for SGPR spilling, it is saved and
; restored. No FP is required.
;
; GCN-LABEL: {{^}}callee_func_sgpr_spill_no_calls:
; GCN: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN: v_writelane_b32 [[CSR_VGPR]], s
; GCN: v_writelane_b32 [[CSR_VGPR]], s

; GCN: ;;#ASMSTART
; GCN: v_readlane_b32 s{{[0-9]+}}, [[CSR_VGPR]]
; GCN: v_readlane_b32 s{{[0-9]+}}, [[CSR_VGPR]]

; GCN: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt
; GCN-NEXT: s_setpc_b64
define void @callee_func_sgpr_spill_no_calls(i32 %in) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}"() #0
  call void asm sideeffect "", "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}"() #0
  call void asm sideeffect "", "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23}"() #0
  call void asm sideeffect "", "~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"() #0
  call void asm sideeffect "", "~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"() #0

  %wide.sgpr0 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr1 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr2 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr5 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr3 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr4 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0

  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr0) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr1) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr2) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr3) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %wide.sgpr4) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr5) #0
  ret void
}

; Has no spilled CSR VGPRs used for SGPR spilling, so no need to
; enable all lanes and restore.

; GCN-LABEL: {{^}}spill_only_csr_sgpr:
; GCN: s_waitcnt
; GCN-NEXT: v_writelane_b32 v0, s42, 0
; GCN-NEXT: ;;#ASMSTART
; GCN-NEXT: ; clobber s42
; GCN-NEXT: ;;#ASMEND
; GCN-NEXT: v_readlane_b32 s42, v0, 0
; GCN-NEXT: s_setpc_b64
define void @spill_only_csr_sgpr() {
  call void asm sideeffect "; clobber s42", "~{s42}"()
  ret void
}

; TODO: Can the SP inc/deec be remvoed?
; GCN-LABEL: {{^}}callee_with_stack_no_fp_elim_csr_vgpr:
; GCN: s_waitcnt
; GCN-NEXT:s_mov_b32 [[FP_COPY:s[0-9]+]], s33
; GCN-NEXT: s_mov_b32 s33, s32
; GCN: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; GCN-DAG: buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; GCN-DAG: buffer_store_dword [[ZERO]], off, s[0:3], s33 offset:8

; GCN:	;;#ASMSTART
; GCN-NEXT: ; clobber v41
; GCN-NEXT: ;;#ASMEND

; GCN: buffer_load_dword v41, off, s[0:3], s33 ; 4-byte Folded Reload
; GCN: s_add_u32 s32, s32, 0x300
; GCN-NEXT: s_sub_u32 s32, s32, 0x300
; GCN-NEXT: s_mov_b32 s33, s4
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @callee_with_stack_no_fp_elim_csr_vgpr() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void asm sideeffect "; clobber v41", "~{v41}"()
  ret void
}

; Use a copy to a free SGPR instead of introducing a second CSR VGPR.
; GCN-LABEL: {{^}}last_lane_vgpr_for_fp_csr:
; GCN: s_waitcnt
; GCN-NEXT: v_writelane_b32 v1, s33, 63
; GCN-NEXT: s_mov_b32 s33, s32
; GCN: buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; GCN-COUNT-63: v_writelane_b32 v1
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33 offset:8
; GCN: ;;#ASMSTART
; GCN-COUNT-63: v_readlane_b32 s{{[0-9]+}}, v1

; GCN: s_add_u32 s32, s32, 0x300
; GCN-NEXT: s_sub_u32 s32, s32, 0x300
; GCN-NEXT: v_readlane_b32 s33, v1, 63
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @last_lane_vgpr_for_fp_csr() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void asm sideeffect "; clobber v41", "~{v41}"()
  call void asm sideeffect "",
    "~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102}"() #1

  ret void
}

; Use a copy to a free SGPR instead of introducing a second CSR VGPR.
; GCN-LABEL: {{^}}no_new_vgpr_for_fp_csr:
; GCN: s_waitcnt
; GCN-NEXT: s_mov_b32 [[FP_COPY:s[0-9]+]], s33
; GCN-NEXT: s_mov_b32 s33, s32
; GCN-NEXT: buffer_store_dword v41, off, s[0:3], s33 ; 4-byte Folded Spill
; GCN-COUNT-64: v_writelane_b32 v1,

; GCN: buffer_store_dword
; GCN: ;;#ASMSTART
; GCN-COUNT-64: v_readlane_b32 s{{[0-9]+}}, v1

; GCN: buffer_load_dword v41, off, s[0:3], s33 ; 4-byte Folded Reload
; GCN: s_add_u32 s32, s32, 0x300
; GCN-NEXT: s_sub_u32 s32, s32, 0x300
; GCN-NEXT: s_mov_b32 s33, [[FP_COPY]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @no_new_vgpr_for_fp_csr() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void asm sideeffect "; clobber v41", "~{v41}"()
  call void asm sideeffect "",
    "~{s39},~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66},~{s67},~{s68},~{s69}
    ,~{s70},~{s71},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79}
    ,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89}
    ,~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98},~{s99}
    ,~{s100},~{s101},~{s102}"() #1

  ret void
}

; GCN-LABEL: {{^}}realign_stack_no_fp_elim:
; GCN: s_waitcnt
; GCN-NEXT: s_add_u32 [[SCRATCH:s[0-9]+]], s32, 0x7ffc0
; GCN-NEXT: s_mov_b32 s4, s33
; GCN-NEXT: s_and_b32 s33, [[SCRATCH]], 0xfff80000
; GCN-NEXT: s_add_u32 s32, s32, 0x100000
; GCN-NEXT: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; GCN-NEXT: buffer_store_dword [[ZERO]], off, s[0:3], s33
; GCN-NEXT: s_sub_u32 s32, s32, 0x100000
; GCN-NEXT: s_mov_b32 s33, s4
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @realign_stack_no_fp_elim() #1 {
  %alloca = alloca i32, align 8192, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}no_unused_non_csr_sgpr_for_fp:
; GCN: s_waitcnt
; GCN-NEXT: v_writelane_b32 v1, s33, 2
; GCN-NEXT: v_writelane_b32 v1, s30, 0
; GCN-NEXT: s_mov_b32 s33, s32
; GCN: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; GCN: v_writelane_b32 v1, s31, 1
; GCN: buffer_store_dword [[ZERO]], off, s[0:3], s33 offset:4
; GCN: ;;#ASMSTART
; GCN: v_readlane_b32 s4, v1, 0
; GCN-NEXT: s_add_u32 s32, s32, 0x200
; GCN-NEXT: v_readlane_b32 s5, v1, 1
; GCN-NEXT: s_sub_u32 s32, s32, 0x200
; GCN-NEXT: v_readlane_b32 s33, v1, 2
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64 s[4:5]
define void @no_unused_non_csr_sgpr_for_fp() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  ; Use all clobberable registers, so FP has to spill to a VGPR.
  call void asm sideeffect "",
    "~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s30},~{s31}"() #0

  ret void
}

; Need a new CSR VGPR to satisfy the FP spill.
; GCN-LABEL: {{^}}no_unused_non_csr_sgpr_for_fp_no_scratch_vgpr:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_store_dword [[CSR_VGPR:v[0-9]+]], off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-NEXT: v_writelane_b32 [[CSR_VGPR]], s33, 2
; GCN-NEXT: v_writelane_b32 [[CSR_VGPR]], s30, 0
; GCN-NEXT: s_mov_b32 s33, s32

; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31, 1
; GCN-DAG: buffer_store_dword
; GCN: s_add_u32 s32, s32, 0x300{{$}}

; GCN: ;;#ASMSTART

; GCN: v_readlane_b32 s4, [[CSR_VGPR]], 0
; GCN-NEXT: v_readlane_b32 s5, [[CSR_VGPR]], 1
; GCN-NEXT: s_sub_u32 s32, s32, 0x300{{$}}
; GCN-NEXT: v_readlane_b32 s33, [[CSR_VGPR]], 2
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: buffer_load_dword [[CSR_VGPR]], off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @no_unused_non_csr_sgpr_for_fp_no_scratch_vgpr() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  ; Use all clobberable registers, so FP has to spill to a VGPR.
  call void asm sideeffect "",
    "~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s30},~{s31}"() #0

  call void asm sideeffect "; clobber nonpreserved initial VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"() #1

  ret void
}

; The byval argument exceeds the MUBUF constant offset, so a scratch
; register is needed to access the CSR VGPR slot.
; GCN-LABEL: {{^}}scratch_reg_needed_mubuf_offset:
; GCN: s_waitcnt
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC0:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: v_mov_b32_e32 [[SCRATCH_VGPR:v[0-9]+]], 0x1008
; GCN-NEXT: buffer_store_dword [[CSR_VGPR:v[0-9]+]], [[SCRATCH_VGPR]], s[0:3], s32 offen ; 4-byte Folded Spill
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC0]]
; GCN-NEXT: v_writelane_b32 [[CSR_VGPR]], s33, 2
; GCN-NEXT: v_writelane_b32 [[CSR_VGPR]], s30, 0
; GCN-NEXT: s_mov_b32 s33, s32
; GCN-DAG: v_writelane_b32 [[CSR_VGPR]], s31, 1
; GCN-DAG: s_add_u32 s32, s32, 0x40300{{$}}
; GCN-DAG: buffer_store_dword

; GCN: ;;#ASMSTART

; GCN: v_readlane_b32 s4, [[CSR_VGPR]], 0
; GCN-NEXT: v_readlane_b32 s5, [[CSR_VGPR]], 1
; GCN-NEXT: s_sub_u32 s32, s32, 0x40300{{$}}
; GCN-NEXT: v_readlane_b32 s33, [[CSR_VGPR]], 2
; GCN-NEXT: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN-NEXT: v_mov_b32_e32 [[SCRATCH_VGPR:v[0-9]+]], 0x1008
; GCN-NEXT: buffer_load_dword [[CSR_VGPR]], [[SCRATCH_VGPR]], s[0:3], s32 offen ; 4-byte Folded Reload
; GCN-NEXT: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @scratch_reg_needed_mubuf_offset([4096 x i8] addrspace(5)* byval align 4 %arg) #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

  ; Use all clobberable registers, so FP has to spill to a VGPR.
  call void asm sideeffect "; clobber nonpreserved SGPRs",
    "~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s30},~{s31}"() #0

  ; Use all clobberable VGPRs, so a CSR spill is needed for the VGPR
  call void asm sideeffect "; clobber nonpreserved VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"() #1

  ret void
}

; GCN-LABEL: {{^}}local_empty_func:
; GCN: s_waitcnt
; GCN-NEXT: s_setpc_b64
define internal void @local_empty_func() #0 {
  ret void
}

; An FP is needed, despite not needing any spills
; TODO: Ccould see callee does not use stack and omit FP.
; GCN-LABEL: {{^}}ipra_call_with_stack:
; GCN: s_mov_b32 [[FP_COPY:s[0-9]+]], s33
; GCN: s_mov_b32 s33, s32
; GCN: s_add_u32 s32, s32, 0x400
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33{{$}}
; GCN: s_swappc_b64
; GCN: s_sub_u32 s32, s32, 0x400
; GCN: s_mov_b32 s33, [[FP_COPY:s[0-9]+]]
define void @ipra_call_with_stack() #0 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void @local_empty_func()
  ret void
}

; With no free registers, we must spill the FP to memory.
; GCN-LABEL: {{^}}callee_need_to_spill_fp_to_memory:
; GCN: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN: v_mov_b32_e32 [[TMP_VGPR1:v[0-9]+]], s33
; GCN: buffer_store_dword [[TMP_VGPR1]], off, s[0:3], s32 offset:[[OFF:[0-9]+]]
; GCN: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN: s_mov_b32 s33, s32
; GCN: s_or_saveexec_b64 [[COPY_EXEC2:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN: buffer_load_dword [[TMP_VGPR2:v[0-9]+]], off, s[0:3], s32 offset:[[OFF]]
; GCN: s_waitcnt vmcnt(0)
; GCN: v_readfirstlane_b32 s33, [[TMP_VGPR2]]
; GCN: s_mov_b64 exec, [[COPY_EXEC2]]
; GCN: s_setpc_b64
define void @callee_need_to_spill_fp_to_memory() #3 {
  call void asm sideeffect "; clobber nonpreserved SGPRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"()
  ret void
}

; If we have a reserved VGPR that can be used for SGPR spills, we may still
; need to spill the FP to memory if there are no free lanes in the reserved
; VGPR.
; GCN-LABEL: {{^}}callee_need_to_spill_fp_to_memory_full_reserved_vgpr:
; GCN: s_or_saveexec_b64 [[COPY_EXEC1:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN: v_mov_b32_e32 [[TMP_VGPR1:v[0-9]+]], s33
; GCN: buffer_store_dword [[TMP_VGPR1]], off, s[0:3], s32 offset:[[OFF:[0-9]+]]
; GCN: s_mov_b64 exec, [[COPY_EXEC1]]
; GCN-NOT: v_writelane_b32 v40, s33
; GCN: s_mov_b32 s33, s32
; GCN-NOT: v_readlane_b32 s33, v40
; GCN: s_or_saveexec_b64 [[COPY_EXEC2:s\[[0-9]+:[0-9]+\]]], -1{{$}}
; GCN: buffer_load_dword [[TMP_VGPR2:v[0-9]+]], off, s[0:3], s32 offset:[[OFF]]
; GCN: v_readfirstlane_b32 s33, [[TMP_VGPR2]]
; GCN: s_mov_b64 exec, [[COPY_EXEC2]]
; GCN: s_setpc_b64
define void @callee_need_to_spill_fp_to_memory_full_reserved_vgpr() #3 {
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
    ,~{s100},~{s101},~{s102},~{s39},~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs except CSR v40",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38}"()
  ret void
}

; If the size of the offset exceeds the MUBUF offset field we need another
; scratch VGPR to hold the offset.
; GCN-LABEL: {{^}}spill_fp_to_memory_scratch_reg_needed_mubuf_offset
; GCN: s_or_saveexec_b64 s[4:5], -1
; GCN: v_mov_b32_e32 v0, s33
; GCN-NOT: v_mov_b32_e32 v0, 0x100c
; GCN-NEXT: v_mov_b32_e32 v1, 0x100c
; GCN-NEXT: buffer_store_dword v0, v1, s[0:3], s32 offen
define void @spill_fp_to_memory_scratch_reg_needed_mubuf_offset([4096 x i8] addrspace(5)* byval align 4 %arg) #3 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca

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
    ,~{s100},~{s101},~{s102},~{s39},~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs except CSR v40",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38}"()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "frame-pointer"="all" }
attributes #2 = { nounwind "frame-pointer"="non-leaf" }
attributes #3 = { nounwind "frame-pointer"="all" "amdgpu-waves-per-eu"="6,6" }
