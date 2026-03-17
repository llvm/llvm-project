; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 < %s | FileCheck -check-prefix=DVGPR %s
; RUN: sed 's/"amdgpu-dynamic-vgpr-block-size"="16"/nounwind/' %s \
; RUN:   | llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 | FileCheck -check-prefix=NODVGPR %s

; DVGPR-DAG:   .set func.0.num_vgpr, 15
; DVGPR-DAG:   .set func.0.has_indirect_call, 1
; DVGPR-DAG:   .set func.1.num_vgpr, 55
; DVGPR-DAG:   .set func.1.has_indirect_call, 1
; DVGPR-DAG:   .set retry_vgpr_alloc.num_vgpr, max(11, amdgpu.max_num_vgpr)
; DVGPR-DAG:   .set retry_vgpr_alloc.has_indirect_call, 1
; DVGPR-DAG:   .set first_retry_wrapper.num_vgpr, max(11, amdgpu.max_num_vgpr)
; DVGPR-DAG:   .set first_retry_wrapper.has_indirect_call, 1
; DVGPR-DAG:   .set amdgpu.max_num_vgpr, 55

; DVGPR:       .shader_functions:
; DVGPR:         func.0:
; DVGPR:           .vgpr_count: 0xf
; DVGPR:         func.1:
; DVGPR:           .vgpr_count: 0x37

; NODVGPR-DAG: .set func.0.num_vgpr, max(15, amdgpu.max_num_vgpr)
; NODVGPR-DAG: .set func.0.has_indirect_call, 1
; NODVGPR-DAG: .set func.1.num_vgpr, max(55, amdgpu.max_num_vgpr)
; NODVGPR-DAG: .set func.1.has_indirect_call, 1
; NODVGPR-DAG: .set retry_vgpr_alloc.num_vgpr, max(11, amdgpu.max_num_vgpr)
; NODVGPR-DAG: .set retry_vgpr_alloc.has_indirect_call, 1
; NODVGPR-DAG: .set first_retry_wrapper.num_vgpr, max(11, amdgpu.max_num_vgpr)
; NODVGPR-DAG: .set first_retry_wrapper.has_indirect_call, 1
; NODVGPR-DAG: .set amdgpu.max_num_vgpr, 55

; NODVGPR:     .shader_functions:
; NODVGPR:       func.0:
; NODVGPR:         .vgpr_count: 0x37
; NODVGPR:       func.1:
; NODVGPR:         .vgpr_count: 0x37

define amdgpu_cs_chain void @func.0(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}"()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> %vgprs, i32 1, i32 0, i32 -1, ptr @first_retry_wrapper)
  unreachable
}

define amdgpu_cs_chain void @func.1(<3 x i32> inreg %sgprs, <3 x i32> %vgprs) #0 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39},~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49}"()
  %fptr = load ptr, ptr inttoptr(i64 0 to ptr)
  call void(ptr, i32, <3 x i32>, <3 x i32>, i32, ...) @llvm.amdgcn.cs.chain.v3i32(ptr inreg %fptr, i32 inreg 0, <3 x i32> inreg %sgprs, <3 x i32> %vgprs, i32 1, i32 0, i32 -1, ptr @first_retry_wrapper)
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
