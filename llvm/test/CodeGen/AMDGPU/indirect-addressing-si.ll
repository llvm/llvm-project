; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MOVREL %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MOVREL %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-vgpr-index-mode -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,IDXMODE %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,IDXMODE %s

; Tests for indirect addressing on SI, which is implemented using dynamic
; indexing of vectors.

; GCN-LABEL: {{^}}extract_w_offset:
; GCN-DAG: s_load_dword [[IN0:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 4.0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40400000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 2.0
; GCN-DAG: v_mov_b32_e32 [[BASEREG:v[0-9]+]], 1.0
; GCN-DAG: s_add_i32 [[IN:s[0-9]+]], [[IN0]], 1

; MOVREL-DAG: s_mov_b32 m0, [[IN]]
; MOVREL: v_movrels_b32_e32 v{{[0-9]+}}, [[BASEREG]]

; IDXMODE: s_set_gpr_idx_on [[IN]], gpr_idx(SRC0){{$}}
; IDXMODE-NEXT: v_mov_b32_e32 v{{[0-9]+}}, [[BASEREG]]
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @extract_w_offset(ptr addrspace(1) %out, i32 %in) {
entry:
  %idx = add i32 %in, 1
  %elt = extractelement <16 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0>, i32 %idx
  store float %elt, ptr addrspace(1) %out
  ret void
}

; XXX: Could do v_or_b32 directly
; GCN-LABEL: {{^}}extract_w_offset_salu_use_vector:
; GCN-DAG: s_or_b32
; GCN-DAG: s_or_b32
; GCN-DAG: s_or_b32
; GCN-DAG: s_or_b32
; MOVREL: s_mov_b32 m0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}


; MOVREL: v_movrels_b32_e32

; IDXMODE: s_set_gpr_idx_on s{{[0-9]+}}, gpr_idx(SRC0){{$}}
; IDXMODE-NEXT: v_mov_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @extract_w_offset_salu_use_vector(ptr addrspace(1) %out, i32 %in, <16 x i32> %or.val) {
entry:
  %idx = add i32 %in, 1
  %vec = or <16 x i32> %or.val, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  %elt = extractelement <16 x i32> %vec, i32 %idx
  store i32 %elt, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}extract_wo_offset:
; GCN-DAG: s_load_dword [[IN:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 4.0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40400000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 2.0
; GCN-DAG: v_mov_b32_e32 [[BASEREG:v[0-9]+]], 1.0

; MOVREL-DAG: s_mov_b32 m0, [[IN]]
; MOVREL: v_movrels_b32_e32 v{{[0-9]+}}, [[BASEREG]]

; IDXMODE: s_set_gpr_idx_on [[IN]], gpr_idx(SRC0){{$}}
; IDXMODE-NEXT: v_mov_b32_e32 v{{[0-9]+}}, [[BASEREG]]
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @extract_wo_offset(ptr addrspace(1) %out, i32 %in) {
entry:
  %elt = extractelement <16 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0>, i32 %in
  store float %elt, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}extract_neg_offset_sgpr:
; The offset depends on the register that holds the first element of the vector.
; MOVREL: s_add_i32 m0, s{{[0-9]+}}, 0xfffffe{{[0-9a-z]+}}
; MOVREL: v_movrels_b32_e32 v{{[0-9]}}, v0

; IDXMODE: s_addk_i32 [[ADD_IDX:s[0-9]+]], 0xfe00{{$}}
; IDXMODE: v_mov_b32_e32 v14, 15
; IDXMODE: v_mov_b32_e32 v15, 16
; IDXMODE-NEXT: s_set_gpr_idx_on [[ADD_IDX]], gpr_idx(SRC0){{$}}
; IDXMODE-NEXT: v_mov_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @extract_neg_offset_sgpr(ptr addrspace(1) %out, i32 %offset) {
entry:
  %index = add i32 %offset, -512
  %value = extractelement <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>, i32 %index
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}extract_neg_offset_sgpr_loaded:
; The offset depends on the register that holds the first element of the vector.
; MOVREL: s_add_i32 m0, s{{[0-9]+}}, 0xfffffe{{[0-9a-z]+}}
; MOVREL: v_movrels_b32_e32 v{{[0-9]}}, v0

; IDXMODE-DAG: s_addk_i32 [[ADD_IDX:s[0-9]+]], 0xfe00{{$}}
; IDXMODE-DAG: v_mov_b32_e32 v0,
; IDXMODE: v_mov_b32_e32 v1,
; IDXMODE: v_mov_b32_e32 v2,
; IDXMODE: v_mov_b32_e32 v3,
; IDXMODE: v_mov_b32_e32 v4,
; IDXMODE: v_mov_b32_e32 v5,
; IDXMODE: v_mov_b32_e32 v6,
; IDXMODE: v_mov_b32_e32 v7,
; IDXMODE: v_mov_b32_e32 v8,
; IDXMODE: v_mov_b32_e32 v9,
; IDXMODE: v_mov_b32_e32 v10,
; IDXMODE: v_mov_b32_e32 v11,
; IDXMODE: v_mov_b32_e32 v12,
; IDXMODE: v_mov_b32_e32 v13,
; IDXMODE: v_mov_b32_e32 v14,
; IDXMODE: v_mov_b32_e32 v15,
; IDXMODE-NEXT: s_set_gpr_idx_on [[ADD_IDX]], gpr_idx(SRC0){{$}}
; IDXMODE-NEXT: v_mov_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @extract_neg_offset_sgpr_loaded(ptr addrspace(1) %out, <16 x i32> %vec0, <16 x i32> %vec1, i32 %offset) {
entry:
  %index = add i32 %offset, -512
  %or = or <16 x i32> %vec0, %vec1
  %value = extractelement <16 x i32> %or, i32 %index
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}extract_neg_offset_vgpr:
; The offset depends on the register that holds the first element of the vector.

; GCN: v_cmp_eq_u32_e32
; GCN-COUNT-14: v_cndmask_b32
; GCN: v_cndmask_b32_e32 [[RESULT:v[0-9]+]], 16
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @extract_neg_offset_vgpr(ptr addrspace(1) %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %index = add i32 %id, -512
  %value = extractelement <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>, i32 %index
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}extract_undef_offset_sgpr:
; undefined behavior, but shouldn't crash compiler
define amdgpu_kernel void @extract_undef_offset_sgpr(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %ld = load volatile <4 x i32>, ptr addrspace(1) %in
  %value = extractelement <4 x i32> %ld, i32 undef
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}insert_undef_offset_sgpr_vector_src:
; undefined behavior, but shouldn't crash compiler
define amdgpu_kernel void @insert_undef_offset_sgpr_vector_src(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %ld = load <4 x i32>, ptr addrspace(1) %in
  %value = insertelement <4 x i32> %ld, i32 5, i32 undef
  store <4 x i32> %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}insert_w_offset:
; GCN-DAG: s_load_dword [[IN0:s[0-9]+]]
; MOVREL-DAG: s_add_i32 [[IN:s[0-9]+]], [[IN0]], 1
; MOVREL-DAG: s_mov_b32 m0, [[IN]]
; GCN-DAG: v_mov_b32_e32 v[[ELT0:[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 v[[ELT1:[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 v[[ELT2:[0-9]+]], 0x40400000
; GCN-DAG: v_mov_b32_e32 v[[ELT3:[0-9]+]], 4.0
; GCN-DAG: v_mov_b32_e32 v[[ELT15:[0-9]+]], 0x41800000
; GCN-DAG: v_mov_b32_e32 v[[INS:[0-9]+]], 0x41880000

; MOVREL: v_movreld_b32_e32 v[[ELT0]], v[[INS]]
; MOVREL: buffer_store_dwordx4 v[[[ELT0]]:[[ELT3]]]
define amdgpu_kernel void @insert_w_offset(ptr addrspace(1) %out, i32 %in) {
entry:
  %add = add i32 %in, 1
  %ins = insertelement <16 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0>, float 17.0, i32 %add
  store <16 x float> %ins, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}insert_unsigned_base_plus_offset:
; GCN-DAG: s_load_dword [[IN:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[ELT0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[ELT1:v[0-9]+]], 2.0
; GCN-DAG: s_and_b32 [[BASE:s[0-9]+]], [[IN]], 0xffff

; MOVREL: s_mov_b32 m0, [[BASE]]
; MOVREL: v_movreld_b32_e32 [[ELT1]], v{{[0-9]+}}

; IDXMODE: s_set_gpr_idx_on [[BASE]], gpr_idx(DST)
; IDXMODE-NEXT: v_mov_b32_e32 [[ELT1]], v{{[0-9]+}}
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @insert_unsigned_base_plus_offset(ptr addrspace(1) %out, i16 %in) {
entry:
  %base = zext i16 %in to i32
  %add = add i32 %base, 1
  %ins = insertelement <16 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0>, float 17.0, i32 %add
  store <16 x float> %ins, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}insert_signed_base_plus_offset:
; GCN-DAG: s_load_dword [[IN:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[ELT0:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 [[ELT1:v[0-9]+]], 2.0

; GCN-DAG: s_sext_i32_i16 [[BASE:s[0-9]+]], [[IN]]
; GCN-DAG: s_add_i32 [[BASE_PLUS_OFFSET:s[0-9]+]], [[BASE]], 1

; MOVREL: s_mov_b32 m0, [[BASE_PLUS_OFFSET]]
; MOVREL: v_movreld_b32_e32 [[ELT0]], v{{[0-9]+}}

; IDXMODE: s_set_gpr_idx_on [[BASE_PLUS_OFFSET]], gpr_idx(DST)
; IDXMODE-NEXT: v_mov_b32_e32 [[ELT0]], v{{[0-9]+}}
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @insert_signed_base_plus_offset(ptr addrspace(1) %out, i16 %in) {
entry:
  %base = sext i16 %in to i32
  %add = add i32 %base, 1
  %ins = insertelement <16 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0>, float 17.0, i32 %add
  store <16 x float> %ins, ptr addrspace(1) %out
  ret void
}


; GCN-LABEL: {{^}}insert_wo_offset:
; GCN: s_load_dword [[IN:s[0-9]+]]

; MOVREL: s_mov_b32 m0, [[IN]]
; MOVREL: v_movreld_b32_e32 v[[ELT0:[0-9]+]]

; IDXMODE: s_set_gpr_idx_on [[IN]], gpr_idx(DST)
; IDXMODE-NEXT: v_mov_b32_e32 v[[ELT0:[0-9]+]], v{{[0-9]+}}
; IDXMODE-NEXT: s_set_gpr_idx_off

; GCN: buffer_store_dwordx4 v[[[ELT0]]:
define amdgpu_kernel void @insert_wo_offset(ptr addrspace(1) %out, i32 %in) {
entry:
  %ins = insertelement <16 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0>, float 17.0, i32 %in
  store <16 x float> %ins, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}insert_neg_offset_sgpr:
; The offset depends on the register that holds the first element of the vector.
; MOVREL: s_add_i32 m0, s{{[0-9]+}}, 0xfffffe{{[0-9a-z]+}}
; MOVREL: v_movreld_b32_e32 v0, 16

; IDXMODE: s_addk_i32 [[ADD_IDX:s[0-9]+]], 0xfe00{{$}}
; IDXMODE: s_set_gpr_idx_on [[ADD_IDX]], gpr_idx(DST)
; IDXMODE-NEXT: v_mov_b32_e32 v0, 16
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @insert_neg_offset_sgpr(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %offset) {
entry:
  %index = add i32 %offset, -512
  %value = insertelement <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, i32 16, i32 %index
  store <16 x i32> %value, ptr addrspace(1) %out
  ret void
}

; The vector indexed into is originally loaded into an SGPR rather
; than built with a reg_sequence

; GCN-LABEL: {{^}}insert_neg_offset_sgpr_loadreg:
; The offset depends on the register that holds the first element of the vector.
; MOVREL: s_add_i32 m0, s{{[0-9]+}}, 0xfffffe{{[0-9a-z]+}}
; MOVREL: v_movreld_b32_e32 v0, 5

; IDXMODE: s_addk_i32 [[ADD_IDX:s[0-9]+]], 0xfe00{{$}}
; IDXMODE: s_set_gpr_idx_on [[ADD_IDX]], gpr_idx(DST)
; IDXMODE-NEXT: v_mov_b32_e32 v0, 5
; IDXMODE-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @insert_neg_offset_sgpr_loadreg(ptr addrspace(1) %in, ptr addrspace(1) %out, <16 x i32> %vec, i32 %offset) {
entry:
  %index = add i32 %offset, -512
  %value = insertelement <16 x i32> %vec, i32 5, i32 %index
  store <16 x i32> %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}insert_neg_offset_vgpr:
; The offset depends on the register that holds the first element of the vector.

; GCN: v_cmp_eq_u32_e32
; GCN-COUNT-16: v_cndmask_b32
; GCN-COUNT-4:  buffer_store_dwordx4
define amdgpu_kernel void @insert_neg_offset_vgpr(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %index = add i32 %id, -512
  %value = insertelement <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>, i32 33, i32 %index
  store <16 x i32> %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}insert_neg_inline_offset_vgpr:

; GCN: v_cmp_eq_u32_e32
; GCN-COUNT-16: v_cndmask_b32
; GCN-COUNT-4:  buffer_store_dwordx4
define amdgpu_kernel void @insert_neg_inline_offset_vgpr(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %index = add i32 %id, -16
  %value = insertelement <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>, i32 500, i32 %index
  store <16 x i32> %value, ptr addrspace(1) %out
  ret void
}

; When the block is split to insert the loop, make sure any other
; places that need to be expanded in the same block are also handled.

; GCN-LABEL: {{^}}extract_vgpr_offset_multiple_in_block:

; GCN-DAG: {{buffer|flat|global}}_load_dword [[IDX0:v[0-9]+]]
; GCN: v_cmp_eq_u32
; GCN: v_cndmask_b32_e64 [[RESULT0:v[0-9]+]], 16,
; GCN: v_cndmask_b32_e64 [[RESULT1:v[0-9]+]], 16,

; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
define amdgpu_kernel void @extract_vgpr_offset_multiple_in_block(ptr addrspace(1) %out0, ptr addrspace(1) %out1, ptr addrspace(1) %in) #0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %id.ext = zext i32 %id to i64
  %gep = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %id.ext
  %idx0 = load volatile i32, ptr addrspace(1) %gep
  %idx1 = add i32 %idx0, 1
  %val0 = extractelement <16 x i32> <i32 7, i32 9, i32 11, i32 13, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>, i32 %idx0
  %live.out.reg = call i32 asm sideeffect "s_mov_b32 $0, 17", "={s4}" ()
  %val1 = extractelement <16 x i32> <i32 7, i32 9, i32 11, i32 13, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>, i32 %idx1
  store volatile i32 %val0, ptr addrspace(1) %out0
  store volatile i32 %val1, ptr addrspace(1) %out0
  %cmp = icmp eq i32 %id, 0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  store volatile i32 %live.out.reg, ptr addrspace(1) undef
  br label %bb2

bb2:
  ret void
}

; Moved subtest for insert_vgpr_offset_multiple_in_block to separate file to
; avoid very different schedule induced isses with gfx9.
; test/CodeGen/AMDGPU/indirect-addressing-si-pregfx9.ll


; GCN-LABEL: {{^}}insert_adjacent_blocks:
define amdgpu_kernel void @insert_adjacent_blocks(i32 %arg, float %val0) #0 {
bb:
  %tmp = icmp eq i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb4

bb1:                                              ; preds = %bb
  %tmp2 = load volatile <4 x float>, ptr addrspace(1) undef
  %tmp3 = insertelement <4 x float> %tmp2, float %val0, i32 undef
  call void asm sideeffect "; reg use $0", "v"(<4 x float> %tmp3) #0 ; Prevent block optimize out
  br label %bb7

bb4:                                              ; preds = %bb
  %tmp5 = load volatile <4 x float>, ptr addrspace(1) undef
  %tmp6 = insertelement <4 x float> %tmp5, float %val0, i32 undef
  call void asm sideeffect "; reg use $0", "v"(<4 x float> %tmp6) #0 ; Prevent block optimize out
  br label %bb7

bb7:                                              ; preds = %bb4, %bb1
  %tmp8 = phi <4 x float> [ %tmp3, %bb1 ], [ %tmp6, %bb4 ]
  store volatile <4 x float> %tmp8, ptr addrspace(1) undef
  ret void
}

; FIXME: Should be able to fold zero input to movreld to inline imm?

; GCN-LABEL: {{^}}multi_same_block:

; GCN: s_load_dword [[ARG:s[0-9]+]]

; MOVREL: v_mov_b32_e32 v{{[0-9]+}}, 0x41900000
; MOVREL: v_mov_b32_e32 v{{[0-9]+}}, 0x41b0cccd
; MOVREL: s_waitcnt
; MOVREL: s_add_i32 m0, [[ARG]], -16
; MOVREL: v_movreld_b32_e32 v{{[0-9]+}}, 4.0
; MOVREL: v_movreld_b32_e32 v{{[0-9]+}}, -4.0
; MOVREL: s_mov_b32 m0, -1


; IDXMODE: v_mov_b32_e32 v{{[0-9]+}}, 0x41900000
; IDXMODE: v_mov_b32_e32 v{{[0-9]+}}, 0x41b0cccd
; IDXMODE: s_waitcnt
; IDXMODE: s_add_i32 [[ARG]], [[ARG]], -16
; IDXMODE: v_mov_b32_e32 v{{[0-9]+}}, 4.0
; IDXMODE: v_mov_b32_e32 v{{[0-9]+}}, -4.0
; IDXMODE: s_set_gpr_idx_off

; GCN: ds_write_b32
; GCN: ds_write_b32
; GCN: s_endpgm
define amdgpu_kernel void @multi_same_block(i32 %arg) #0 {
bb:
  %tmp1 = add i32 %arg, -16
  %tmp2 = insertelement <9 x float> <float 1.700000e+01, float 1.800000e+01, float 1.900000e+01, float 2.000000e+01, float 2.100000e+01, float 2.200000e+01, float 2.300000e+01, float 2.400000e+01, float 2.500000e+01>, float 4.000000e+00, i32 %tmp1
  %tmp3 = add i32 %arg, -16
  %tmp4 = insertelement <9 x float> <float 0x40311999A0000000, float 0x40321999A0000000, float 0x40331999A0000000, float 0x40341999A0000000, float 0x40351999A0000000, float 0x40361999A0000000, float 0x40371999A0000000, float 0x40381999A0000000, float 0x40391999A0000000>, float -4.0, i32 %tmp3
  %tmp5 = bitcast <9 x float> %tmp2 to <9 x i32>
  %tmp6 = extractelement <9 x i32> %tmp5, i32 1
  %tmp7 = bitcast <9 x float> %tmp4 to <9 x i32>
  %tmp8 = extractelement <9 x i32> %tmp7, i32 5
  store volatile i32 %tmp6, ptr addrspace(3) undef, align 4
  store volatile i32 %tmp8, ptr addrspace(3) undef, align 4
  ret void
}

; offset puts outside of superegister bounaries, so clamp to 1st element.
; GCN-LABEL: {{^}}extract_largest_inbounds_offset:
; GCN-DAG: buffer_load_dwordx4 v[[[LO_ELT:[0-9]+]]:[[HI_ELT:[0-9]+]]
; GCN-DAG: s_load_dword [[IDX0:s[0-9]+]]
; GCN-DAG: s_add_i32 [[IDX:s[0-9]+]], [[IDX0]], 15

; MOVREL: s_mov_b32 m0, [[IDX]]
; MOVREL: v_movrels_b32_e32 [[EXTRACT:v[0-9]+]], v[[LO_ELT]]

; IDXMODE: s_set_gpr_idx_on [[IDX]], gpr_idx(SRC0)
; IDXMODE: v_mov_b32_e32 [[EXTRACT:v[0-9]+]], v[[LO_ELT]]
; IDXMODE: s_set_gpr_idx_off

; GCN: buffer_store_dword [[EXTRACT]]
define amdgpu_kernel void @extract_largest_inbounds_offset(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx) {
entry:
  %ld = load volatile <16 x i32>, ptr addrspace(1) %in
  %offset = add i32 %idx, 15
  %value = extractelement <16 x i32> %ld, i32 %offset
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}extract_out_of_bounds_offset:
; GCN-DAG: buffer_load_dwordx4 v[[[LO_ELT:[0-9]+]]:[[HI_ELT:[0-9]+]]]
; GCN-DAG: s_load_dword [[IDX:s[0-9]+]]
; GCN: s_add_i32 [[ADD_IDX:s[0-9]+]], [[IDX]], 16

; MOVREL: s_mov_b32 m0, [[ADD_IDX]]
; MOVREL: v_movrels_b32_e32 [[EXTRACT:v[0-9]+]], v[[LO_ELT]]

; IDXMODE: s_set_gpr_idx_on [[ADD_IDX]], gpr_idx(SRC0)
; IDXMODE: v_mov_b32_e32 [[EXTRACT:v[0-9]+]], v[[LO_ELT]]
; IDXMODE: s_set_gpr_idx_off

; GCN: buffer_store_dword [[EXTRACT]]
define amdgpu_kernel void @extract_out_of_bounds_offset(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx) {
entry:
  %ld = load volatile <16 x i32>, ptr addrspace(1) %in
  %offset = add i32 %idx, 16
  %value = extractelement <16 x i32> %ld, i32 %offset
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}extractelement_v16i32_or_index:
; GCN: s_load_dword [[IDX_IN:s[0-9]+]]
; GCN: s_lshl_b32 [[IDX_SHL:s[0-9]+]], [[IDX_IN]]

; MOVREL: s_mov_b32 m0, [[IDX_SHL]]
; MOVREL: v_movrels_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}

; IDXMODE: s_set_gpr_idx_on [[IDX_SHL]], gpr_idx(SRC0)
; IDXMODE: v_mov_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; IDXMODE: s_set_gpr_idx_off
define amdgpu_kernel void @extractelement_v16i32_or_index(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %idx.in) {
entry:
  %ld = load volatile <16 x i32>, ptr addrspace(1) %in
  %idx.shl = shl i32 %idx.in, 2
  %idx = or i32 %idx.shl, 1
  %value = extractelement <16 x i32> %ld, i32 %idx
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}insertelement_v16f32_or_index:
; GCN: s_load_dword [[IDX_IN:s[0-9]+]]
; GCN: s_lshl_b32 [[IDX_SHL:s[0-9]+]], [[IDX_IN]]

; MOVREL: s_mov_b32 m0, [[IDX_SHL]]
; MOVREL: v_movreld_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}

; IDXMODE: s_set_gpr_idx_on [[IDX_SHL]], gpr_idx(DST)
; IDXMODE: v_mov_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; IDXMODE: s_set_gpr_idx_off
define amdgpu_kernel void @insertelement_v16f32_or_index(ptr addrspace(1) %out, <16 x float> %a, i32 %idx.in) nounwind {
  %idx.shl = shl i32 %idx.in, 2
  %idx = or i32 %idx.shl, 1
  %vecins = insertelement <16 x float> %a, float 5.000000e+00, i32 %idx
  store <16 x float> %vecins, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}broken_phi_bb:
; GCN: v_mov_b32_e32 [[PHIREG:v[0-9]+]], 8

; GCN: {{.LBB[0-9]+_[0-9]+}}:
; GCN: [[BB2:.LBB[0-9]+_[0-9]+]]:
; GCN: v_cmp_le_i32_e32 vcc, s{{[0-9]+}}, [[PHIREG]]
; GCN: buffer_load_dword

; GCN: [[REGLOOP:.LBB[0-9]+_[0-9]+]]:
; MOVREL: v_movreld_b32_e32

; IDXMODE: s_set_gpr_idx_on
; IDXMODE: v_mov_b32_e32
; IDXMODE: s_set_gpr_idx_off

; GCN: s_cbranch_execnz [[REGLOOP]]

; GCN: {{^; %bb.[0-9]}}:
; GCN: s_mov_b64 exec,
; GCN: s_cbranch_execnz [[BB2]]

define amdgpu_kernel void @broken_phi_bb(i32 %arg, i32 %arg1) #0 {
bb:
  br label %bb2

bb2:                                              ; preds = %bb4, %bb
  %tmp = phi i32 [ 8, %bb ], [ %tmp7, %bb4 ]
  %tmp3 = icmp slt i32 %tmp, %arg
  br i1 %tmp3, label %bb4, label %bb8

bb4:                                              ; preds = %bb2
  %vgpr = load volatile i32, ptr addrspace(1) undef
  %tmp5 = insertelement <16 x i32> undef, i32 undef, i32 %vgpr
  %tmp6 = insertelement <16 x i32> %tmp5, i32 %arg1, i32 %vgpr
  %tmp7 = extractelement <16 x i32> %tmp6, i32 0
  br label %bb2

bb8:                                              ; preds = %bb2
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare void @llvm.amdgcn.s.barrier() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind convergent }
