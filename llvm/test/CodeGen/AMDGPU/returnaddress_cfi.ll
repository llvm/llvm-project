; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -O0 -verify-machineinstrs < %s | FileCheck %s


; Function Attrs: convergent mustprogress nounwind
define hidden void @_ZL3barv_spill_RA_to_vgpr() #0 {
; CHECK-LABEL: _ZL3barv_spill_RA_to_vgpr:
; CHECK:       .Lfunc_begin0:
; CHECK-NEXT:    .cfi_sections .debug_frame
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:  ; %bb.0: ; %entry
; CHECK-NEXT:    .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT:    .cfi_escape 0x10, 0x10, 0x08, 0x90, 0x3e, 0x93, 0x04, 0x90, 0x3f, 0x93, 0x04 ;
; CHECK:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    s_or_saveexec_b64 s[16:17], -1
; CHECK-NEXT:    buffer_store_dword v40, off, s[0:3], s32 offset:192 ; 4-byte Folded Spill
; CHECK-NEXT:    .cfi_offset 2600, 1228
; CHECK-NEXT:    s_mov_b64 exec, s[16:17]

; CHECK:    v_writelane_b32 v40, s30, 32
; CHECK-NEXT:    v_writelane_b32 v40, s31, 33
; CHECK-NEXT:    .cfi_escape 0x10, 0x10, 0x0e, 0x90, 0xa8, 0x14, 0x9d, 0x20, 0x80, 0x08, 0x90, 0xa8, 0x14, 0x9d, 0x20, 0xa0, 0x08 ;
; CHECK:    ;;#ASMSTART
; CHECK-NEXT:    ; clobber nonpreserved and 32 CSR SGPRs
; CHECK-NEXT:    ;;#ASMEND

; CHECK:    ;;#ASMSTART
; CHECK-NEXT:    ; clobber all VGPRs except v40
; CHECK-NEXT:    ;;#ASMEND
; CHECK:    s_getpc_b64 s[16:17]
; CHECK-NEXT:    s_add_u32 s16, s16, _ZL13sleep_foreverv@gotpcrel32@lo+4
; CHECK-NEXT:    s_addc_u32 s17, s17, _ZL13sleep_foreverv@gotpcrel32@hi+12
; CHECK:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    s_swappc_b64 s[30:31], s[16:17]

; CHECK-DAG:    v_readlane_b32 s30, v40, 32
; CHECK-DAG:    v_readlane_b32 s31, v40, 33

; CHECK:    s_or_saveexec_b64 s[4:5], -1
; CHECK-NEXT:    buffer_load_dword v40, off, s[0:3], s32 offset:192 ; 4-byte Folded Reload
; CHECK-NEXT:    s_mov_b64 exec, s[4:5]
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_setpc_b64 s[30:31]
entry:
  call void asm sideeffect "; clobber nonpreserved and 32 CSR SGPRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s34},~{s35},~{s36},~{s37},~{s38},~{s39}
    ,~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65}
    ,~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs except v40",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}
    ,~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49}
    ,~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},~{v56},~{v57},~{v58},~{v59}
    ,~{v60},~{v61},~{v62},~{v63},~{v64},~{v65},~{v66},~{v67},~{v68},~{v69}
    ,~{v70},~{v71},~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79}
    ,~{v80},~{v81},~{v82},~{v83},~{v84},~{v85},~{v86},~{v87},~{v88},~{v89}
    ,~{v90},~{v91},~{v92},~{v93},~{v94},~{v95},~{v96},~{v97},~{v98},~{v99}
    ,~{v100},~{v101},~{v102},~{v103},~{v104},~{v105},~{v106},~{v107},~{v108},~{v109}
    ,~{v110},~{v111},~{v112},~{v113},~{v114},~{v115},~{v116},~{v117},~{v118},~{v119}
    ,~{v120},~{v121},~{v122},~{v123},~{v124},~{v125},~{v126},~{v127},~{v128},~{v129}"()

  call void @_ZL13sleep_foreverv()
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define hidden void @_ZL3barv_spill_RA_to_memory() #0 {
; CHECK-LABEL: _ZL3barv_spill_RA_to_memory:
; CHECK:       .Lfunc_begin1:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:  ; %bb.0: ; %entry
; CHECK-NEXT:    .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT:    .cfi_escape 0x10, 0x10, 0x08, 0x90, 0x3e, 0x93, 0x04, 0x90, 0x3f, 0x93, 0x04 ;
; CHECK:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_mov_b32_e32 v0, s33
; CHECK-NEXT:    buffer_store_dword v0, off, s[0:3], s32 offset:380 ; 4-byte Folded Spill
; CHECK-NEXT:    .cfi_offset 65, 24320
; CHECK-NEXT:    s_mov_b32 s33, s32
; CHECK-NEXT:    .cfi_def_cfa_register 65
; CHECK-NEXT:    s_add_i32 s32, s32, 0x6400

; CHECK:    s_waitcnt vmcnt(0)
; CHECK:    s_mov_b64 exec, s[16:17]
; CHECK:    s_mov_b64 s[16:17], exec
; CHECK:    s_mov_b64 exec, 3
; CHECK-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:384
; CHECK-NEXT:    v_writelane_b32 v0, s30, 0
; CHECK-NEXT:    v_writelane_b32 v0, s31, 1
; CHECK-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:372 ; 4-byte Folded Spill
; CHECK-NEXT:    .cfi_offset 16, 23808
; CHECK-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:384

; CHECK:    ;;#ASMSTART
; CHECK-NEXT:    ; clobber nonpreserved and 32 CSR SGPRs
; CHECK-NEXT:    ;;#ASMEND

; CHECK:    ;;#ASMSTART
; CHECK-NEXT:    ; clobber all VGPRs
; CHECK-NEXT:    ;;#ASMEND

; CHECK:    s_getpc_b64 s[16:17]
; CHECK-NEXT:    s_add_u32 s16, s16, _ZL13sleep_foreverv@gotpcrel32@lo+4
; CHECK-NEXT:    s_addc_u32 s17, s17, _ZL13sleep_foreverv@gotpcrel32@hi+12
; CHECK-NEXT:    s_load_dwordx2 s[16:17], s[16:17], 0x0
; CHECK:    s_waitcnt lgkmcnt(0)
; CHECK-NEXT:    s_swappc_b64 s[30:31], s[16:17]

; CHECK-NEXT:    s_mov_b64 s[4:5], exec
; CHECK-NEXT:    s_mov_b64 exec, 3
; CHECK-NEXT:    buffer_store_dword v0, off, s[0:3], s33 offset:384
; CHECK-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:372 ; 4-byte Folded Reload
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    v_readlane_b32 s30, v0, 0
; CHECK-NEXT:    v_readlane_b32 s31, v0, 1
; CHECK-NEXT:    buffer_load_dword v0, off, s[0:3], s33 offset:384
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    s_mov_b64 exec, s[4:5]

; CHECK:    s_add_i32 s32, s32, 0xffff9c00
; CHECK-NEXT:    buffer_load_dword v0, off, s[0:3], s32 offset:380 ; 4-byte Folded Reload
; CHECK-NEXT:    s_waitcnt vmcnt(0)
; CHECK-NEXT:    v_readfirstlane_b32 s33, v0
; CHECK-NEXT:    .cfi_def_cfa_register 64
; CHECK-NEXT:    s_setpc_b64 s[30:31]
entry:
  call void asm sideeffect "; clobber nonpreserved and 32 CSR SGPRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{s34},~{s35},~{s36},~{s37},~{s38},~{s39}
    ,~{s40},~{s41},~{s42},~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49}
    ,~{s50},~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58},~{s59}
    ,~{s60},~{s61},~{s62},~{s63},~{s64},~{s65}
    ,~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}
    ,~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49}
    ,~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},~{v56},~{v57},~{v58},~{v59}
    ,~{v60},~{v61},~{v62},~{v63},~{v64},~{v65},~{v66},~{v67},~{v68},~{v69}
    ,~{v70},~{v71},~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79}
    ,~{v80},~{v81},~{v82},~{v83},~{v84},~{v85},~{v86},~{v87},~{v88},~{v89}
    ,~{v90},~{v91},~{v92},~{v93},~{v94},~{v95},~{v96},~{v97},~{v98},~{v99}
    ,~{v100},~{v101},~{v102},~{v103},~{v104},~{v105},~{v106},~{v107},~{v108},~{v109}
    ,~{v110},~{v111},~{v112},~{v113},~{v114},~{v115},~{v116},~{v117},~{v118},~{v119}
    ,~{v120},~{v121},~{v122},~{v123},~{v124},~{v125},~{v126},~{v127},~{v128},~{v129}"()

  call void @_ZL13sleep_foreverv()
  ret void
}

; Function Attrs: convergent nounwind
declare void @_ZL13sleep_foreverv() #0

attributes #0 = { nounwind "frame-pointer"="all" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1638, !1639, !1640, !1641}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !1, producer: "clang version 13.0.0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "lane-info.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "4ab9b75a30baffdf0f6f536a80e3e382")
!371 = !DISubroutineType(types: !372)
!372 = !{null}
!1638 = !{i32 7, !"Dwarf Version", i32 5}
!1639 = !{i32 2, !"Debug Info Version", i32 3}
!1640 = !{i32 1, !"wchar_size", i32 4}
!1641 = !{i32 7, !"PIC Level", i32 1}
