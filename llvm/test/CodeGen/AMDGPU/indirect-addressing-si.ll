; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; Tests for indirect addressing on SI, which is implemented using dynamic
; indexing of vectors.

; CHECK-LABEL: {{^}}extract_w_offset:
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, 4.0
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40400000
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, 2.0
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, 1.0
; CHECK: s_mov_b32 m0
; CHECK-NEXT: v_movrels_b32_e32
define void @extract_w_offset(float addrspace(1)* %out, i32 %in) {
entry:
  %idx = add i32 %in, 1
  %elt = extractelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 %idx
  store float %elt, float addrspace(1)* %out
  ret void
}

; XXX: Could do v_or_b32 directly
; CHECK-LABEL: {{^}}extract_w_offset_salu_use_vector:
; CHECK-DAG: s_or_b32
; CHECK-DAG: s_or_b32
; CHECK-DAG: s_or_b32
; CHECK-DAG: s_or_b32
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; CHECK: s_mov_b32 m0
; CHECK-NEXT: v_movrels_b32_e32
define void @extract_w_offset_salu_use_vector(i32 addrspace(1)* %out, i32 %in, <4 x i32> %or.val) {
entry:
  %idx = add i32 %in, 1
  %vec = or <4 x i32> %or.val, <i32 1, i32 2, i32 3, i32 4>
  %elt = extractelement <4 x i32> %vec, i32 %idx
  store i32 %elt, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}extract_wo_offset:
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, 4.0
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40400000
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, 2.0
; CHECK-DAG: v_mov_b32_e32 v{{[0-9]+}}, 1.0
; CHECK: s_mov_b32 m0
; CHECK-NEXT: v_movrels_b32_e32
define void @extract_wo_offset(float addrspace(1)* %out, i32 %in) {
entry:
  %elt = extractelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 %in
  store float %elt, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}extract_neg_offset_sgpr:
; The offset depends on the register that holds the first element of the vector.
; CHECK: s_add_i32 m0, s{{[0-9]+}}, 0xfffffe{{[0-9a-z]+}}
; CHECK: v_movrels_b32_e32 v{{[0-9]}}, v0
define void @extract_neg_offset_sgpr(i32 addrspace(1)* %out, i32 %offset) {
entry:
  %index = add i32 %offset, -512
  %value = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}extract_neg_offset_sgpr_loaded:
; The offset depends on the register that holds the first element of the vector.
; CHECK: s_add_i32 m0, s{{[0-9]+}}, 0xfffffe{{[0-9a-z]+}}
; CHECK: v_movrels_b32_e32 v{{[0-9]}}, v0
define void @extract_neg_offset_sgpr_loaded(i32 addrspace(1)* %out, <4 x i32> %vec0, <4 x i32> %vec1, i32 %offset) {
entry:
  %index = add i32 %offset, -512
  %or = or <4 x i32> %vec0, %vec1
  %value = extractelement <4 x i32> %or, i32 %index
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}extract_neg_offset_vgpr:
; The offset depends on the register that holds the first element of the vector.
; CHECK: v_readfirstlane_b32
; CHECK: s_add_i32 m0, m0, 0xfffffe{{[0-9a-z]+}}
; CHECK-NEXT: v_movrels_b32_e32 v{{[0-9]}}, v0
; CHECK: s_cbranch_execnz
define void @extract_neg_offset_vgpr(i32 addrspace(1)* %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %index = add i32 %id, -512
  %value = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}extract_undef_offset_sgpr:
define void @extract_undef_offset_sgpr(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
entry:
  %ld = load volatile <4 x i32>, <4  x i32> addrspace(1)* %in
  %value = extractelement <4 x i32> %ld, i32 undef
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}insert_undef_offset_sgpr_vector_src:
; CHECK: buffer_load_dwordx4
; CHECK: s_mov_b32 m0,
; CHECK-NEXT: v_movreld_b32
define void @insert_undef_offset_sgpr_vector_src(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
entry:
  %ld = load <4 x i32>, <4  x i32> addrspace(1)* %in
  %value = insertelement <4 x i32> %ld, i32 5, i32 undef
  store <4 x i32> %value, <4 x i32> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}insert_w_offset:
; CHECK: s_mov_b32 m0
; CHECK-NEXT: v_movreld_b32_e32
define void @insert_w_offset(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = add i32 %in, 1
  %1 = insertelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, float 5.0, i32 %0
  %2 = extractelement <4 x float> %1, i32 2
  store float %2, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}insert_wo_offset:
; CHECK: s_mov_b32 m0
; CHECK-NEXT: v_movreld_b32_e32
define void @insert_wo_offset(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = insertelement <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, float 5.0, i32 %in
  %1 = extractelement <4 x float> %0, i32 2
  store float %1, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}insert_neg_offset_sgpr:
; The offset depends on the register that holds the first element of the vector.
; CHECK: s_add_i32 m0, s{{[0-9]+}}, 0xfffffe{{[0-9a-z]+}}
; CHECK: v_movreld_b32_e32 v0, v{{[0-9]}}
define void @insert_neg_offset_sgpr(i32 addrspace(1)* %in, <4 x i32> addrspace(1)* %out, i32 %offset) {
entry:
  %index = add i32 %offset, -512
  %value = insertelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 5, i32 %index
  store <4 x i32> %value, <4 x i32> addrspace(1)* %out
  ret void
}

; The vector indexed into is originally loaded into an SGPR rather
; than built with a reg_sequence

; CHECK-LABEL: {{^}}insert_neg_offset_sgpr_loadreg:
; The offset depends on the register that holds the first element of the vector.
; CHECK: s_add_i32 m0, s{{[0-9]+}}, 0xfffffe{{[0-9a-z]+}}
; CHECK: v_movreld_b32_e32 v0, v{{[0-9]}}
define void @insert_neg_offset_sgpr_loadreg(i32 addrspace(1)* %in, <4 x i32> addrspace(1)* %out, <4 x i32> %vec, i32 %offset) {
entry:
  %index = add i32 %offset, -512
  %value = insertelement <4 x i32> %vec, i32 5, i32 %index
  store <4 x i32> %value, <4 x i32> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}insert_neg_offset_vgpr:
; The offset depends on the register that holds the first element of the vector.
; CHECK: v_readfirstlane_b32
; CHECK: s_add_i32 m0, m0, 0xfffffe{{[0-9a-z]+}}
; CHECK-NEXT: v_movreld_b32_e32 v0, v{{[0-9]}}
; CHECK: s_cbranch_execnz
define void @insert_neg_offset_vgpr(i32 addrspace(1)* %in, <4 x i32> addrspace(1)* %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %index = add i32 %id, -512
  %value = insertelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 5, i32 %index
  store <4 x i32> %value, <4 x i32> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}insert_neg_inline_offset_vgpr:
; The offset depends on the register that holds the first element of the vector.
; CHECK: v_readfirstlane_b32
; CHECK: s_add_i32 m0, m0, -{{[0-9]+}}
; CHECK-NEXT: v_movreld_b32_e32 v0, v{{[0-9]}}
; CHECK: s_cbranch_execnz
define void @insert_neg_inline_offset_vgpr(i32 addrspace(1)* %in, <4 x i32> addrspace(1)* %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %index = add i32 %id, -16
  %value = insertelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 5, i32 %index
  store <4 x i32> %value, <4 x i32> addrspace(1)* %out
  ret void
}

; When the block is split to insert the loop, make sure any other
; places that need to be expanded in the same block are also handled.

; CHECK-LABEL: {{^}}extract_vgpr_offset_multiple_in_block:

; CHECK-DAG: {{buffer|flat}}_load_dword [[IDX0:v[0-9]+]]
; CHECK-DAG: s_mov_b32 [[S_ELT0:s[0-9]+]], 7
; CHECK-DAG: s_mov_b32 [[S_ELT1:s[0-9]+]], 9
; CHECK-DAG: v_mov_b32_e32 [[VEC_ELT0:v[0-9]+]], [[S_ELT0]]
; CHECK-DAG: v_mov_b32_e32 [[VEC_ELT1:v[0-9]+]], [[S_ELT1]]
; CHECK: s_waitcnt vmcnt(0)

; CHECK: s_mov_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], exec

; CHECK: [[LOOP0:BB[0-9]+_[0-9]+]]:
; CHECK: v_readfirstlane_b32 vcc_lo, [[IDX0]]
; CHECK: s_mov_b32 m0, vcc_lo
; CHECK: v_cmp_eq_u32_e32 vcc, m0, [[IDX0]]
; CHECK: s_and_saveexec_b64 vcc, vcc
; CHECK-NEXT: v_movrels_b32_e32 [[MOVREL0:v[0-9]+]], [[VEC_ELT0]]
; CHECK-NEXT: s_xor_b64 exec, exec, vcc
; CHECK: s_cbranch_execnz [[LOOP0]]

; FIXME: Redundant copy
; CHECK: s_mov_b64 exec, [[MASK]]
; CHECK: s_mov_b64 [[MASK2:s\[[0-9]+:[0-9]+\]]], exec

; CHECK: [[LOOP1:BB[0-9]+_[0-9]+]]:
; CHECK: v_readfirstlane_b32 vcc_lo, [[IDX0]]
; CHECK: s_mov_b32 m0, vcc_lo
; CHECK: v_cmp_eq_u32_e32 vcc, m0, [[IDX0]]
; CHECK: s_and_saveexec_b64 vcc, vcc
; CHECK-NEXT: v_movrels_b32_e32 [[MOVREL1:v[0-9]+]], [[VEC_ELT1]]
; CHECK-NEXT: s_xor_b64 exec, exec, vcc
; CHECK: s_cbranch_execnz [[LOOP1]]

; CHECK: buffer_store_dword [[MOVREL0]]
; CHECK: buffer_store_dword [[MOVREL1]]
define void @extract_vgpr_offset_multiple_in_block(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, i32 addrspace(1)* %in) #0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %id.ext = zext i32 %id to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %id.ext
  %idx0 = load volatile i32, i32 addrspace(1)* %gep
  %idx1 = add i32 %idx0, 1
  %val0 = extractelement <4 x i32> <i32 7, i32 9, i32 11, i32 13>, i32 %idx0
  %live.out.reg = call i32 asm sideeffect "s_mov_b32 $0, 17", "={SGPR4}" ()
  %val1 = extractelement <4 x i32> <i32 7, i32 9, i32 11, i32 13>, i32 %idx1
  store volatile i32 %val0, i32 addrspace(1)* %out0
  store volatile i32 %val1, i32 addrspace(1)* %out0
  %cmp = icmp eq i32 %id, 0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  store volatile i32 %live.out.reg, i32 addrspace(1)* undef
  br label %bb2

bb2:
  ret void
}

; CHECK-LABEL: {{^}}insert_vgpr_offset_multiple_in_block:
; CHECK-DAG: s_load_dwordx4 s{{\[}}[[S_ELT0:[0-9]+]]:[[S_ELT3:[0-9]+]]{{\]}}
; CHECK-DAG: {{buffer|flat}}_load_dword [[IDX0:v[0-9]+]]
; CHECK-DAG: v_mov_b32_e32 [[VEC_ELT0:v[0-9]+]], s[[S_ELT0]]
; CHECK-DAG: v_mov_b32 [[INS0:v[0-9]+]], 62
; CHECK-DAG: s_waitcnt vmcnt(0)

; CHECK: s_mov_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], exec

; CHECK: [[LOOP0:BB[0-9]+_[0-9]+]]:
; CHECK: v_readfirstlane_b32 vcc_lo, [[IDX0]]
; CHECK: s_mov_b32 m0, vcc_lo
; CHECK: v_cmp_eq_u32_e32 vcc, m0, [[IDX0]]
; CHECK: s_and_saveexec_b64 vcc, vcc
; CHECK-NEXT: v_movreld_b32_e32 v[[MOVREL0:[0-9]+]], [[INS0]]
; CHECK-NEXT: s_xor_b64 exec, exec, vcc
; CHECK: s_cbranch_execnz [[LOOP0]]

; FIXME: Redundant copy
; CHECK: s_mov_b64 exec, [[MASK]]
; CHECK: v_mov_b32_e32 [[INS1:v[0-9]+]], 63
; CHECK: s_mov_b64 [[MASK]], exec

; CHECK: [[LOOP1:BB[0-9]+_[0-9]+]]:
; CHECK: v_readfirstlane_b32 vcc_lo, [[IDX0]]
; CHECK: s_mov_b32 m0, vcc_lo
; CHECK: v_cmp_eq_u32_e32 vcc, m0, [[IDX0]]
; CHECK: s_and_saveexec_b64 vcc, vcc
; CHECK-NEXT: v_movreld_b32_e32 v[[MOVREL1:[0-9]+]], [[INS1]]
; CHECK-NEXT: s_xor_b64 exec, exec, vcc
; CHECK: s_cbranch_execnz [[LOOP1]]

; CHECK: buffer_store_dwordx4 v{{\[}}[[MOVREL0]]:

; CHECK: buffer_store_dword [[INS0]]
define void @insert_vgpr_offset_multiple_in_block(<4 x i32> addrspace(1)* %out0, <4 x i32> addrspace(1)* %out1, i32 addrspace(1)* %in, <4 x i32> %vec0) #0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %id.ext = zext i32 %id to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %id.ext
  %idx0 = load volatile i32, i32 addrspace(1)* %gep
  %idx1 = add i32 %idx0, 1
  %live.out.val = call i32 asm sideeffect "v_mov_b32 $0, 62", "=v"()
  %vec1 = insertelement <4 x i32> %vec0, i32 %live.out.val, i32 %idx0
  %vec2 = insertelement <4 x i32> %vec1, i32 63, i32 %idx1
  store volatile <4 x i32> %vec2, <4 x i32> addrspace(1)* %out0
  %cmp = icmp eq i32 %id, 0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  store volatile i32 %live.out.val, i32 addrspace(1)* undef
  br label %bb2

bb2:
  ret void
}

; CHECK-LABEL: {{^}}extract_adjacent_blocks:
; CHECK: s_load_dword [[ARG:s[0-9]+]]
; CHECK: s_cmp_lg_i32
; CHECK: s_cbranch_scc0 [[BB4:BB[0-9]+_[0-9]+]]

; CHECK: buffer_load_dwordx4
; CHECK: s_mov_b32 m0,
; CHECK: v_movrels_b32_e32
; CHECK: s_branch [[ENDBB:BB[0-9]+_[0-9]+]]

; CHECK: [[BB4]]:
; CHECK: buffer_load_dwordx4
; CHECK: s_mov_b32 m0,
; CHECK: v_movrels_b32_e32

; CHECK: [[ENDBB]]:
; CHECK: buffer_store_dword
; CHECK: s_endpgm
define void @extract_adjacent_blocks(i32 %arg) #0 {
bb:
  %tmp = icmp eq i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb4

bb1:
  %tmp2 = load volatile <4 x float>, <4 x float> addrspace(1)* undef
  %tmp3 = extractelement <4 x float> %tmp2, i32 undef
  br label %bb7

bb4:
  %tmp5 = load volatile <4 x float>, <4 x float> addrspace(1)* undef
  %tmp6 = extractelement <4 x float> %tmp5, i32 undef
  br label %bb7

bb7:
  %tmp8 = phi float [ %tmp3, %bb1 ], [ %tmp6, %bb4 ]
  store volatile float %tmp8, float addrspace(1)* undef
  ret void
}

; CHECK-LABEL: {{^}}insert_adjacent_blocks:
; CHECK: s_load_dword [[ARG:s[0-9]+]]
; CHECK: s_cmp_lg_i32
; CHECK: s_cbranch_scc0 [[BB4:BB[0-9]+_[0-9]+]]

; CHECK: buffer_load_dwordx4
; CHECK: s_mov_b32 m0,
; CHECK: v_movreld_b32_e32
; CHECK: s_branch [[ENDBB:BB[0-9]+_[0-9]+]]

; CHECK: [[BB4]]:
; CHECK: buffer_load_dwordx4
; CHECK: s_mov_b32 m0,
; CHECK: v_movreld_b32_e32

; CHECK: [[ENDBB]]:
; CHECK: buffer_store_dword
; CHECK: s_endpgm
define void @insert_adjacent_blocks(i32 %arg, float %val0) #0 {
bb:
  %tmp = icmp eq i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb4

bb1:                                              ; preds = %bb
  %tmp2 = load volatile <4 x float>, <4 x float> addrspace(1)* undef
  %tmp3 = insertelement <4 x float> %tmp2, float %val0, i32 undef
  br label %bb7

bb4:                                              ; preds = %bb
  %tmp5 = load volatile <4 x float>, <4 x float> addrspace(1)* undef
  %tmp6 = insertelement <4 x float> %tmp5, float %val0, i32 undef
  br label %bb7

bb7:                                              ; preds = %bb4, %bb1
  %tmp8 = phi <4 x float> [ %tmp3, %bb1 ], [ %tmp6, %bb4 ]
  store volatile <4 x float> %tmp8, <4 x float> addrspace(1)* undef
  ret void
}

; FIXME: Should be able to fold zero input to movreld to inline imm?

; CHECK-LABEL: {{^}}multi_same_block:
; CHECK: s_load_dword [[ARG:s[0-9]+]]
; CHECK-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; CHECK-DAG: s_add_i32 m0, [[ARG]], -16
; CHECK: v_movreld_b32_e32 v{{[0-9]+}}, [[ZERO]]

; CHECK: s_add_i32 m0, [[ARG]], -14
; CHECK: v_movreld_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}

; CHECK: s_mov_b32 m0, -1
; CHECK: ds_write_b32
; CHECK: ds_write_b32
; CHECK: s_endpgm
define void @multi_same_block(i32 %arg) #0 {
bb:
  %tmp1 = add i32 %arg, -16
  %tmp2 = insertelement <6 x float> <float 1.700000e+01, float 1.800000e+01, float 1.900000e+01, float 2.000000e+01, float 2.100000e+01, float 2.200000e+01>, float 0.000000e+00, i32 %tmp1
  %tmp3 = add i32 %arg, -16
  %tmp4 = insertelement <6 x float> <float 0x40311999A0000000, float 0x40321999A0000000, float 0x40331999A0000000, float 0x40341999A0000000, float 0x40351999A0000000, float 0x40361999A0000000>, float 0x3FB99999A0000000, i32 %tmp3
  %tmp5 = bitcast <6 x float> %tmp2 to <6 x i32>
  %tmp6 = extractelement <6 x i32> %tmp5, i32 1
  %tmp7 = bitcast <6 x float> %tmp4 to <6 x i32>
  %tmp8 = extractelement <6 x i32> %tmp7, i32 5
  store volatile i32 %tmp6, i32 addrspace(3)* undef, align 4
  store volatile i32 %tmp8, i32 addrspace(3)* undef, align 4
  ret void
}

; offset puts outside of superegister bounaries, so clamp to 1st element.
; CHECK-LABEL: {{^}}extract_largest_inbounds_offset:
; CHECK: buffer_load_dwordx4 v{{\[}}[[LO_ELT:[0-9]+]]:[[HI_ELT:[0-9]+]]{{\]}}
; CHECK: s_load_dword [[IDX:s[0-9]+]]
; CHECK: s_mov_b32 m0, [[IDX]]
; CHECK-NEXT: v_movrels_b32_e32 [[EXTRACT:v[0-9]+]], v[[HI_ELT]]
; CHECK: buffer_store_dword [[EXTRACT]]
define void @extract_largest_inbounds_offset(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in, i32 %idx) {
entry:
  %ld = load volatile <4 x i32>, <4  x i32> addrspace(1)* %in
  %offset = add i32 %idx, 3
  %value = extractelement <4 x i32> %ld, i32 %offset
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABL: {{^}}extract_out_of_bounds_offset:
; CHECK: buffer_load_dwordx4 v{{\[}}[[LO_ELT:[0-9]+]]:[[HI_ELT:[0-9]+]]{{\]}}
; CHECK: s_load_dword [[IDX:s[0-9]+]]
; CHECK: s_add_i32 m0, [[IDX]], 4
; CHECK-NEXT: v_movrels_b32_e32 [[EXTRACT:v[0-9]+]], v[[LO_ELT]]
; CHECK: buffer_store_dword [[EXTRACT]]
define void @extract_out_of_bounds_offset(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in, i32 %idx) {
entry:
  %ld = load volatile <4 x i32>, <4  x i32> addrspace(1)* %in
  %offset = add i32 %idx, 4
  %value = extractelement <4 x i32> %ld, i32 %offset
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
