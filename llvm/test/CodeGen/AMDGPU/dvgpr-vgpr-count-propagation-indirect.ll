; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 < %s | FileCheck -check-prefix=DVGPR %s
; RUN: sed 's/"amdgpu-dynamic-vgpr-block-size"="16"/"amdgpu-dynamic-vgpr-block-size"="0"/' %s \
; RUN:   | llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 | FileCheck -check-prefix=NODVGPR %s

; In DVGPR mode, chain functions that have indirect non-chain calls should have
; their `num_vgpr` count set to the max of their local count and the module-wide
; max VGPR count of all non-chain functions.

; DVGPR:  .set .Lgfx_func_a.num_vgpr, 40
; DVGPR:  .set .Lgfx_func_b.num_vgpr, 80
; DVGPR:  .set .Lfunc_with_indirect_call.num_vgpr, max(11, amdgpu.max_num_vgpr)
; DVGPR:  .set .Lfunc_direct_only.num_vgpr, max(11, .Lgfx_func_a.num_vgpr)
; DVGPR:  .set .Lfunc_chain_only.num_vgpr, 11
; DVGPR:  .set amdgpu.max_num_vgpr, 80

; NODVGPR:  .set .Lgfx_func_a.num_vgpr, 40
; NODVGPR:  .set .Lgfx_func_b.num_vgpr, 80
; NODVGPR:  .set .Lfunc_with_indirect_call.num_vgpr, min(192, max(11, amdgpu.max_num_vgpr))
; NODVGPR:  .set .Lfunc_direct_only.num_vgpr, min(192, max(11, amdgpu.max_num_vgpr))
; NODVGPR:  .set .Lfunc_chain_only.num_vgpr, min(192, max(11, amdgpu.max_num_vgpr))
; NODVGPR:  .set amdgpu.max_num_vgpr, 80

define amdgpu_gfx void @gfx_func_a() #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"()
  ret void
}

define amdgpu_gfx void @gfx_func_b() #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49},~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63},~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71},~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79}"()
  ret void
}

define amdgpu_cs_chain void @func_with_indirect_call(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2}"()
  call amdgpu_gfx void @gfx_func_a()
  %fptr_gfx = load ptr, ptr inttoptr(i64 8 to ptr)
  call amdgpu_gfx void %fptr_gfx()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @func_chain_only)
  unreachable
}

define amdgpu_cs_chain void @func_direct_only(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3}"()
  call amdgpu_gfx void @gfx_func_a()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @func_chain_only)
  unreachable
}

define amdgpu_cs_chain void @func_chain_only(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4}"()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @func_chain_only)
  unreachable
}

declare void @llvm.amdgcn.cs.chain.v3i32(ptr, i32, <3 x i32>, <3 x i32>, i32 immarg, ...)
attributes #0 = { "amdgpu-dynamic-vgpr-block-size"="16" }
