; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-spill-cfi-saved-regs -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,WAVE64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -amdgpu-spill-cfi-saved-regs -verify-machineinstrs -o - %s | FileCheck --check-prefixes=CHECK,WAVE32 %s

; CHECK-LABEL: kern:
; CHECK: .cfi_startproc
; CHECK-NOT: .cfi_{{.*}}
; CHECK: %bb.0:
; CHECK-NEXT: .cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02
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
; CHECK-NEXT: .cfi_llvm_register_pair 16, 62, 32, 63, 32

; WAVE64: v_writelane_b32 v0, exec_lo, 0
; WAVE64-NEXT: v_writelane_b32 v0, exec_hi, 1
; WAVE64-NEXT: .cfi_llvm_vector_registers 17, {{[0-9]+}}, {{[0-9]+}}, 32, {{[0-9]+}}, {{[0-9]+}}, 32

; WAVE32: v_writelane_b32 v0, exec_lo, 0
; WAVE32-NEXT: .cfi_llvm_vector_registers 1, {{[0-9]+}}, {{[0-9]+}}, 32

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

; There's no return here, so the return address live in was deleted.
; CHECK-LABEL: {{^}}empty_func:
; CHECK-NOT: v_writelane_b32 v0, s30, 0
; CHECK-NOT: v_writelane_b32 v0, s31, 1
define void @empty_func() {
  unreachable
}

; Check that the option causes RA and EXEC to be spilled to memory.

; CHECK-LABEL: no_vgprs_to_spill_into:
; CHECK: %bb.0:

; WAVE64: v_mov_b32_e32 v0, exec_lo
; WAVE64-NEXT: buffer_store_dword v0, off, s[0:3], s32 ; 4-byte Folded Spill
; WAVE64-NEXT: v_mov_b32_e32 v0, exec_hi
; WAVE64-NEXT: buffer_store_dword v0, off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; WAVE64-NEXT: .cfi_offset 17, 0
 
define void @no_vgprs_to_spill_into() #1 {
  call void asm sideeffect "",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24}"()

  ret void
}

; Check that the FP and EXEC needs to be spilled to memory, even though
; we have reserved VGPR but there are no available free lanes.

; CHECK-LABEL: callee_need_to_spill_fp_exec_to_memory:
; CHECK: %bb.0:

; WAVE32: s_mov_b32 [[FP_SCRATCH_COPY:s[0-9]+]], s33
; WAVE32: s_xor_saveexec_b32 [[EXEC_COPY:s[0-9]+]], -1
; WAVE32-NEXT: buffer_store_dword [[RES_VGPR:v[0-9]+]], off, s[0:3], s33 offset:192 ; 4-byte Folded Spill
; WAVE32: s_mov_b32 exec_lo, [[EXEC_COPY]]
; WAVE32-NEXT: v_mov_b32_e32 [[TEMP_VGPR:v[0-9]+]], exec_lo
; WAVE32-NEXT: buffer_store_dword [[TEMP_VGPR]], off, s[0:3], s33 offset:196 ; 4-byte Folded Spill
; WAVE32-NEXT: .cfi_offset 1, 6272
; WAVE32-NEXT: v_mov_b32_e32 [[TEMP_VGPR:v[0-9]+]], [[FP_SCRATCH_COPY]]
; WAVE32-NEXT: buffer_store_dword [[TEMP_VGPR]], off, s[0:3], s33 offset:200 ; 4-byte Folded Spill
; WAVE32: buffer_store_dword v40, off, s[0:3], s33 offset
; WAVE32-COUNT-47: buffer_store_dword v{{[0-9]+}}, off, s[0:3], s33
; WAVE32: v_writelane_b32 [[RES_VGPR]], s34, 0
; WAVE32-COUNT-31: v_writelane_b32 [[RES_VGPR]], s{{[0-9]+}}, {{[0-9]+}}


define void @callee_need_to_spill_fp_exec_to_memory() #2 {
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

define internal void @caller_needs_to_spill_pc_to_memory() #3 {
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
  ,~{v120},~{v121},~{v122},~{v123},~{v124},~{v125},~{v126},~{v127},~{v128},~{v129}
  ,~{v130},~{v131},~{v132},~{v133},~{v134},~{v135},~{v136},~{v137},~{v138},~{v139}
  ,~{v140},~{v141},~{v142},~{v143},~{v144},~{v145},~{v146},~{v147},~{v148},~{v149}
  ,~{v150},~{v151},~{v152},~{v153},~{v154},~{v155},~{v156},~{v157},~{v158},~{v159}
  ,~{v160},~{v161},~{v162},~{v163},~{v164},~{v165},~{v166},~{v167},~{v168},~{v169}
  ,~{v170},~{v171},~{v172},~{v173},~{v174},~{v175},~{v176},~{v177},~{v178},~{v179}
  ,~{v180},~{v181},~{v182},~{v183},~{v184},~{v185},~{v186},~{v187},~{v188},~{v189}
  ,~{v190},~{v191},~{v192},~{v193},~{v194},~{v195},~{v196},~{v197},~{v198},~{v199}
  ,~{v200},~{v201},~{v202},~{v203},~{v204},~{v205},~{v206},~{v207},~{v208},~{v209}
  ,~{v210},~{v211},~{v212},~{v213},~{v214},~{v215},~{v216},~{v217},~{v218},~{v219}
  ,~{v220},~{v221},~{v222},~{v223},~{v224},~{v225},~{v226},~{v227},~{v228},~{v229}
  ,~{v230},~{v231},~{v232},~{v233},~{v234},~{v235},~{v236},~{v237},~{v238},~{v239}
  ,~{v240},~{v241},~{v242},~{v243},~{v244},~{v245},~{v246},~{v247},~{v248},~{v249}
  ,~{v250},~{v251},~{v252},~{v253},~{v254},~{v255}" () #3
  ret void
}

; WAVE64-LABEL: need_to_spill_pc_to_mem:
; WAVE64:      s_mov_b64 exec, 3
; WAVE64-NEXT: buffer_store_dword [[TEMP_VGPR:v[0-9]+]]
; WAVE64-NEXT: v_writelane_b32 [[TEMP_VGPR]], s30, 0
; WAVE64-NEXT: v_writelane_b32 [[TEMP_VGPR]], s31, 1
; WAVE64-NEXT: buffer_store_dword [[TEMP_VGPR]], off, s[0:3], s33 offset:
; WAVE64-NEXT: .cfi_offset 16,
; WAVE64-NEXT: buffer_load_dword [[TEMP_VGPR]]

; WAVE32-LABEL: need_to_spill_pc_to_mem:
; WAVE32:      s_mov_b32 exec_lo, 3
; WAVE32-NEXT: buffer_store_dword [[TEMP_VGPR:v[0-9]+]]
; WAVE32-NEXT: v_writelane_b32 [[TEMP_VGPR]], s30, 0
; WAVE32-NEXT: v_writelane_b32 [[TEMP_VGPR]], s31, 1
; WAVE32-NEXT: buffer_store_dword [[TEMP_VGPR]], off, s[0:3], s33 offset:
; WAVE32-NEXT: .cfi_offset 16,
; WAVE32-NEXT: buffer_load_dword [[TEMP_VGPR]]

define void @need_to_spill_pc_to_mem() #3 {
  call void @caller_needs_to_spill_pc_to_memory()
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-waves-per-eu"="10,10" }
attributes #2 = { nounwind "frame-pointer"="all" "amdgpu-waves-per-eu"="12,12" }
attributes #3 = { nounwind norecurse }


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "filename", directory: "directory")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
