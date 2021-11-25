; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-spill-cfi-saved-regs -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,WAVE64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -amdgpu-spill-cfi-saved-regs -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,WAVE32 %s

; CHECK-LABEL: kern:
; CHECK: .cfi_startproc
; CHECK-NOT: .cfi_{{.*}}
; CHECK: %bb.0:
; CHECK-NEXT: .cfi_escape 0x0f, 0x03, 0x30, 0x36, 0xe1
; CHECK-NEXT: .cfi_undefined 16
; CHECK-NOT: .cfi_{{.*}}
; CHECK: .cfi_endproc
define protected amdgpu_kernel void @kern() #0 {
entry:
  ret void
}

; CHECK-LABEL: func_saved_in_clobbered_vgpr:
; CHECK: .cfi_startproc
; CHECK-NOT: .cfi_{{.*}}
; CHECK: %bb.0:
; SGPR32 = 64
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT: .cfi_escape 0x10, 0x10, 0x08, 0x90, 0x3e, 0x93, 0x04, 0x90, 0x3f, 0x93, 0x04


; FIXME: ideally this would not care what VGPR we spill to, but since we are
; using .cfi_escape it isn't trivial/possible to make this general yet

; CHECK: v_writelane_b32 v0, s30, 0
; CHECK-NEXT: v_writelane_b32 v0, s31, 1

; DW_CFA_expression [0x10]
;   PC_64 ULEB128(17)=[0x10]
;   BLOCK_LENGTH ULEB128(12)=[0x0c]
;     DW_OP_regx [0x90]
;       VGPR0_wave64 ULEB128(2560)=[0x80, 0x14]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x00]
;     DW_OP_regx [0x90]
;       VGPR0_wave64 ULEB128(2560)=[0x80, 0x14]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x20]
; WAVE64-NEXT: .cfi_escape 0x10, 0x10, 0x0c, 0x90, 0x80, 0x14, 0x9d, 0x20, 0x00, 0x90, 0x80, 0x14, 0x9d, 0x20, 0x20

; DW_CFA_expression [0x10]
;   PC_64 ULEB128(17)=[0x10]
;   BLOCK_LENGTH ULEB128(12)=[0x0c]
;     DW_OP_regx [0x90]
;       VGPR0_wave32 ULEB128(1536)=[0x80, 0x0c]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x00]
;     DW_OP_regx [0x90]
;       VGPR0_wave32 ULEB128(1536)=[0x80, 0x0c]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x20]
; WAVE32-NEXT: .cfi_escape 0x10, 0x10, 0x0c, 0x90, 0x80, 0x0c, 0x9d, 0x20, 0x00, 0x90, 0x80, 0x0c, 0x9d, 0x20, 0x20


; WAVE64: v_writelane_b32 v0, exec_lo, 2
; WAVE64-NEXT: v_writelane_b32 v0, exec_hi, 3
; DW_CFA_expression [0x10]
;   EXEC_MASK_wave64 ULEB128(17)=[0x11]
;   BLOCK_LENGTH ULEB128(12)=[0x0c]
;     DW_OP_regx [0x90]
;       VGPR0_wave64 ULEB128(2560)=[0x80, 0x14]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x40]
;     DW_OP_regx [0x90]
;       VGPR0_wave64 ULEB128(2560)=[0x80, 0x14]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x60]
; WAVE64-NEXT: .cfi_escape 0x10, 0x11, 0x0c, 0x90, 0x80, 0x14, 0x9d, 0x20, 0x40, 0x90, 0x80, 0x14, 0x9d, 0x20, 0x60

; WAVE32: v_writelane_b32 v0, exec_lo, 2
; DW_CFA_expression [0x10]
;   EXEC_MASK_wave32 ULEB128(1)=[0x01]
;   BLOCK_LENGTH ULEB128(6)=[0x06]
;     DW_OP_regx [0x90]
;       VGPR0_wave32 ULEB128(1536)=[0x80, 0x0c]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x40]
; WAVE32-NEXT: .cfi_escape 0x10, 0x01, 0x06, 0x90, 0x80, 0x0c, 0x9d, 0x20, 0x40

; CHECK-NOT: .cfi_{{.*}}
; CHECK: .cfi_endproc
define hidden void @func_saved_in_clobbered_vgpr() #0 {
entry:
  ret void
}

; Check that the option causes a CSR VGPR to spill when needed.

; CHECK-LABEL: func_saved_in_preserved_vgpr:
; CHECK: %bb.0:

; CHECK: s_or_saveexec_b{{(32|64)}}
; CHECK: buffer_store_dword [[CSR:v[0-9]+]], off, s[0:3], s32 ; 4-byte Folded Spill
; CHECK: s_mov_b{{(32|64)}} {{(exec|exec_lo)}},

; CHECK: v_writelane_b32 [[CSR]], s30, {{[0-9]+}}
; CHECK-NEXT: v_writelane_b32 [[CSR]], s31, {{[0-9]+}}

; WAVE64: v_writelane_b32 [[CSR]], exec_lo, {{[0-9]+}}
; WAVE64-NEXT: v_writelane_b32 [[CSR]], exec_hi, {{[0-9]+}}

; WAVE32: v_writelane_b32 [[CSR]], exec_lo, {{[0-9]+}}

define hidden void @func_saved_in_preserved_vgpr() #0 {
entry:
  call void asm sideeffect "; clobber nonpreserved VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"()
  ret void
}

; There's no return here, so the return address live in was
; deleted. It needs to be re-added as a live in to the entry block.
; CHECK-LABEL: {{^}}empty_func:
; CHECK: v_writelane_b32 v0, s30, 0
; CHECK: v_writelane_b32 v0, s31, 1
define void @empty_func() {
  unreachable
}

; Check that the option causes RA and EXEC to be spilled to memory.

; CHECK-LABEL: no_vgprs_to_spill_into:
; CHECK: %bb.0:

; WAVE64: v_mov_b32_e32 v0, s30
; WAVE64-NEXT: buffer_store_dword v0, off, s[0:3], s32 ; 4-byte Folded Spill
; WAVE64-NEXT: v_mov_b32_e32 v0, s31
; WAVE64-NEXT: buffer_store_dword v0, off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; WAVE64-NEXT: .cfi_offset 16, 0
; WAVE64-NEXT: v_mov_b32_e32 v0, exec_lo
; WAVE64-NEXT: buffer_store_dword v0, off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; WAVE64-NEXT: v_mov_b32_e32 v0, exec_hi
; WAVE64-NEXT: buffer_store_dword v0, off, s[0:3], s32 offset:12 ; 4-byte Folded Spill
; WAVE64-NEXT: .cfi_offset 17, 512
 
define void @no_vgprs_to_spill_into() #1 {
  call void asm sideeffect "",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24}"()

  ret void
}

; Check that the FP, RA and EXEC needs to be spilled to memory, even though
; we have reserved VGPR but there are no available free lanes.

; CHECK-LABEL: callee_need_to_spill_fp_ra_exec_to_memory:
; CHECK: %bb.0:

; WAVE32: s_or_saveexec_b32 [[EXEC_COPY:s[0-9]+]], -1
; WAVE32-NEXT: buffer_store_dword [[RES_VGPR:v[0-9]+]], off, s[0:3], s32 offset:192 ; 4-byte Folded Spill
; WAVE32: s_mov_b32 exec_lo, [[EXEC_COPY]]
; WAVE32: v_mov_b32_e32 [[TEMP_VGPR:v[0-9]+]], s30
; WAVE32-NEXT: buffer_store_dword [[TEMP_VGPR]], off, s[0:3], s32 offset:196 ; 4-byte Folded Spill
; WAVE32-NEXT: v_mov_b32_e32 [[TEMP_VGPR]], s31
; WAVE32-NEXT: buffer_store_dword [[TEMP_VGPR]], off, s[0:3], s32 offset:200 ; 4-byte Folded Spill
; WAVE32-NEXT: .cfi_offset 16, 6272
; WAVE32-NEXT: v_mov_b32_e32 [[TEMP_VGPR]], exec_lo
; WAVE32-NEXT: buffer_store_dword [[TEMP_VGPR]], off, s[0:3], s32 offset:204 ; 4-byte Folded Spill
; WAVE32-NEXT: .cfi_offset 1, 6528
; WAVE32-NEXT: v_mov_b32_e32 [[TEMP_VGPR]], s33
; WAVE32-NEXT: buffer_store_dword [[TEMP_VGPR]], off, s[0:3], s32 offset:208 ; 4-byte Folded Spill
; WAVE32: buffer_store_dword v40, off, s[0:3], s33 offset
; WAVE32-COUNT-47: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33
; WAVE32: v_writelane_b32 [[RES_VGPR]], s34, 0
; WAVE32-COUNT-31: v_writelane_b32 [[RES_VGPR]], s{{[0-9]+}}, {{[0-9]+}}


define void @callee_need_to_spill_fp_ra_exec_to_memory() #2 {
  call void asm sideeffect "; clobber nonpreserved and 32 CSR SGPRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s34},~{s35},~{s36},~{s37},~{s38},~{s39}
    ,~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65}
    ,~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs except v39",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38}
    ,~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49}
    ,~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},~{v56},~{v57},~{v58},~{v59}
    ,~{v60},~{v61},~{v62},~{v63},~{v64},~{v65},~{v66},~{v67},~{v68},~{v69}
    ,~{v70},~{v71},~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79}
    ,~{v80},~{v81},~{v82},~{v83},~{v84},~{v85},~{v86},~{v87},~{v88},~{v89}
    ,~{v90},~{v91},~{v92},~{v93},~{v94},~{v95},~{v96},~{v97},~{v98},~{v99}
    ,~{v100},~{v101},~{v102},~{v103},~{v104},~{v105},~{v106},~{v107},~{v108},~{v109}
    ,~{v110},~{v111},~{v112},~{v113},~{v114},~{v115},~{v116},~{v117},~{v118},~{v119}
    ,~{v120},~{v121},~{v122},~{v123},~{v124},~{v125},~{v126},~{v127},~{v128},~{v129}"()
  ret void
}


attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-waves-per-eu"="10,10" }
attributes #2 = { nounwind "frame-pointer"="all" "amdgpu-waves-per-eu"="12,12" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "filename", directory: "directory")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
