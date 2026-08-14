; RUN: llc -mtriple=amdgpu12.00-amd-amdpal < %s | FileCheck -check-prefix=DVGPR %s
; RUN: sed 's/"amdgpu-dynamic-vgpr-block-size"="16"/"amdgpu-dynamic-vgpr-block-size"="0"/' %s \
; RUN:   | llc -mtriple=amdgpu12.00-amd-amdpal | FileCheck -check-prefix=NODVGPR %s

; Chain calls are treated as indirect calls, but with dynamic VGPRs enabled each
; function will get its own VGPR allocation. Chain functions that have no
; indirect non-chain calls should have their num_vgpr count set to the max of
; their local count and the counts of all direct callees.

; With DVGPRs enabled "max_num_vgpr" represents the maximum VGPR count of all
; non-chain functions. With DVGPRs disabled, it represents module-wide max VGPR
; count.

; DVGPR:  .set .Lgfx_func_a.num_vgpr, 40
; DVGPR:  .set .Lgfx_func_b2.num_vgpr, 80
; DVGPR:  .set .Lgfx_func_b.num_vgpr, max(61, .Lgfx_func_b2.num_vgpr)
; DVGPR:  .set .Lamdgpu_cs_main.num_vgpr, max(42, .Lgfx_func_a.num_vgpr)
; DVGPR:  .set .Lfunc.0.num_vgpr, 13
; DVGPR:  .set .Lfunc.1.num_vgpr, max(14, .Lgfx_func_a.num_vgpr, .Lgfx_func_b.num_vgpr)
; DVGPR:  .set .Lfunc.2.num_vgpr, max(16, .Lgfx_func_a.num_vgpr)
; DVGPR:  .set .Lfunc.3.num_vgpr, max(15, .Lgfx_func_b.num_vgpr)
; DVGPR:  .set .Lfunc.4.num_vgpr, max(100, .Lgfx_func_b.num_vgpr)
; DVGPR:  .set .Lretry_vgpr_alloc.num_vgpr, 11
; DVGPR:  .set .Lfirst_retry_wrapper.num_vgpr, 11
; DVGPR:  .set amdgpu.max_num_vgpr, 80

; NODVGPR:  .set .Lgfx_func_a.num_vgpr, 40
; NODVGPR:  .set .Lgfx_func_b2.num_vgpr, 80
; NODVGPR:  .set .Lgfx_func_b.num_vgpr, max(61, .Lgfx_func_b2.num_vgpr)
; NODVGPR:  .set .Lamdgpu_cs_main.num_vgpr, max(42, amdgpu.max_num_vgpr)
; NODVGPR:  .set .Lfunc.0.num_vgpr, max(13, amdgpu.max_num_vgpr)
; NODVGPR:  .set .Lfunc.1.num_vgpr, max(14, amdgpu.max_num_vgpr)
; NODVGPR:  .set .Lfunc.2.num_vgpr, max(16, amdgpu.max_num_vgpr)
; NODVGPR:  .set .Lfunc.3.num_vgpr, max(15, amdgpu.max_num_vgpr)
; NODVGPR:  .set .Lfunc.4.num_vgpr, max(100, amdgpu.max_num_vgpr)
; NODVGPR:  .set .Lretry_vgpr_alloc.num_vgpr, max(11, amdgpu.max_num_vgpr)
; NODVGPR:  .set .Lfirst_retry_wrapper.num_vgpr, max(11, amdgpu.max_num_vgpr)
; NODVGPR:  .set amdgpu.max_num_vgpr, 100

; DVGPR:  - .hardware_stages:
; DVGPR:        .vgpr_count: 0x2a
; DVGPR:    .shader_functions:
; DVGPR:      func.0:
; DVGPR:        .vgpr_count: 0xd
; DVGPR:      func.1:
; DVGPR:        .vgpr_count: 0x50
; DVGPR:      func.2:
; DVGPR:        .vgpr_count: 0x28
; DVGPR:      func.3:
; DVGPR:        .vgpr_count: 0x50
; DVGPR:      func.4:
; DVGPR:        .vgpr_count: 0x64
; DVGPR:      gfx_func_a:
; DVGPR:        .vgpr_count: 0x28
; DVGPR:      gfx_func_b:
; DVGPR:        .vgpr_count: 0x50
; DVGPR:      gfx_func_b2:
; DVGPR:        .vgpr_count: 0x50

; NODVGPR:  - .hardware_stages:
; NODVGPR:        .vgpr_count: 0x64
; NODVGPR:    .shader_functions:
; NODVGPR:      func.0:
; NODVGPR:        .vgpr_count: 0x64
; NODVGPR:      func.1:
; NODVGPR:        .vgpr_count: 0x64
; NODVGPR:      func.2:
; NODVGPR:        .vgpr_count: 0x64
; NODVGPR:      func.3:
; NODVGPR:        .vgpr_count: 0x64
; NODVGPR:      func.4:
; NODVGPR:        .vgpr_count: 0x64
; NODVGPR:      gfx_func_a:
; NODVGPR:        .vgpr_count: 0x28
; NODVGPR:      gfx_func_b:
; NODVGPR:        .vgpr_count: 0x50
; NODVGPR:      gfx_func_b2:
; NODVGPR:        .vgpr_count: 0x50

define amdgpu_gfx void @gfx_func_a() #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}"()
  ret void
}

define amdgpu_gfx void @gfx_func_b2() #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49},~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63},~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71},~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79}"()
  ret void
}

define amdgpu_gfx void @gfx_func_b() #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49},~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},~{v56},~{v57},~{v58},~{v59}"()
  call amdgpu_gfx void @gfx_func_b2()
  ret void
}

define amdgpu_cs void @amdgpu_cs_main(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call amdgpu_gfx void @gfx_func_a()
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @func.1)
  unreachable
}

define amdgpu_cs_chain void @func.0(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10}"()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @first_retry_wrapper)
  unreachable
}

define amdgpu_cs_chain void @func.1(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13}"()
  call amdgpu_gfx void @gfx_func_a()
  call amdgpu_gfx void @gfx_func_b()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @first_retry_wrapper)
  unreachable
}

define amdgpu_cs_chain void @func.2(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}"()
  call amdgpu_gfx void @gfx_func_a()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @first_retry_wrapper)
  unreachable
}

define amdgpu_cs_chain void @func.3(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14}"()
  call amdgpu_gfx void @gfx_func_b()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @first_retry_wrapper)
  unreachable
}

define amdgpu_cs_chain void @func.4(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49},~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},~{v56},~{v57},~{v58},~{v59},~{v60},~{v61},~{v62},~{v63},~{v64},~{v65},~{v66},~{v67},~{v68},~{v69},~{v70},~{v71},~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79},~{v80},~{v81},~{v82},~{v83},~{v84},~{v85},~{v86},~{v87},~{v88},~{v89},~{v90},~{v91},~{v92},~{v93},~{v94},~{v95},~{v96},~{v97},~{v98},~{v99}"()
  call amdgpu_gfx void @gfx_func_b()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @first_retry_wrapper)
  unreachable
}

define amdgpu_cs_chain_preserve void @retry_vgpr_alloc(<3 x i32> inreg %sgprs) #0 {
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @retry_vgpr_alloc)
  unreachable
}

define amdgpu_cs_chain_preserve void @first_retry_wrapper(<3 x i32> inreg %sgprs) #0 {
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg @retry_vgpr_alloc, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> zeroinitializer, i32 1, i32 0, i32 -1, ptr @retry_vgpr_alloc)
  unreachable
}

attributes #0 = { "amdgpu-dynamic-vgpr-block-size"="16" }
