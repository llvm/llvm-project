; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9PLUS,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9PLUS,GFX10 %s

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_v2i16:
; GFX9PLUS: v_pk_add_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; FIXME: or should be unnecessary
; VI: v_add_u16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_add_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_or_b32
define amdgpu_kernel void @v_test_add_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0, <2 x i16> addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in1, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %b = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in1
  %add = add <2 x i16> %a, %b
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_add_v2i16:
; GFX9PLUS: s_load_dword [[VAL0:s[0-9]+]]
; GFX9PLUS: s_load_dword [[VAL1:s[0-9]+]]
; GFX9: v_mov_b32_e32 [[VVAL1:v[0-9]+]]
; GFX9: v_pk_add_u16 v{{[0-9]+}}, [[VAL1]], [[VVAL1]]
; GFX10: v_pk_add_u16 v{{[0-9]+}}, [[VAL0]], [[VAL1]]

; VI: s_add_i32
; VI: s_add_i32
define amdgpu_kernel void @s_test_add_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %in0, <2 x i16> addrspace(4)* %in1) #1 {
  %a = load <2 x i16>, <2 x i16> addrspace(4)* %in0
  %b = load <2 x i16>, <2 x i16> addrspace(4)* %in1
  %add = add <2 x i16> %a, %b
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_add_self_v2i16:
; GFX9PLUS: s_load_dword [[VAL:s[0-9]+]]
; GFX9PLUS: v_pk_add_u16 v{{[0-9]+}}, [[VAL]], [[VAL]]

; VI: s_add_i32
; VI: s_add_i32
define amdgpu_kernel void @s_test_add_self_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %in0) #1 {
  %a = load <2 x i16>, <2 x i16> addrspace(4)* %in0
  %add = add <2 x i16> %a, %a
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; FIXME: VI should not scalarize arg access.
; GCN-LABEL: {{^}}s_test_add_v2i16_kernarg:
; GFX9: v_pk_add_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GFX10: v_pk_add_u16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}

; VI: s_add_i32
; VI: s_add_i32
; VI: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; VI: s_and_b32
; VI: s_or_b32
define amdgpu_kernel void @s_test_add_v2i16_kernarg(<2 x i16> addrspace(1)* %out, <2 x i16> %a, <2 x i16> %b) #1 {
  %add = add <2 x i16> %a, %b
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; FIXME: Eliminate or with sdwa
; GCN-LABEL: {{^}}v_test_add_v2i16_constant:
; GFX9: s_mov_b32 [[CONST:s[0-9]+]], 0x1c8007b{{$}}
; GFX9: v_pk_add_u16 v{{[0-9]+}}, v{{[0-9]+}}, [[CONST]]
; GFX10: v_pk_add_u16 v{{[0-9]+}}, 0x1c8007b, v{{[0-9]+}}

; VI-DAG: v_mov_b32_e32 v[[SCONST:[0-9]+]], 0x1c8
; VI-DAG: v_add_u16_e32 v{{[0-9]+}}, 0x7b, v{{[0-9]+}}
; VI-DAG: v_add_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v[[SCONST]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI: v_or_b32_e32
define amdgpu_kernel void @v_test_add_v2i16_constant(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %add = add <2 x i16> %a, <i16 123, i16 456>
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_v2i16_neg_constant:
; GFX9: s_mov_b32 [[CONST:s[0-9]+]], 0xfc21fcb3{{$}}
; GFX9: v_pk_add_u16 v{{[0-9]+}}, v{{[0-9]+}}, [[CONST]]
; GFX10: v_pk_add_u16 v{{[0-9]+}}, 0xfc21fcb3, v{{[0-9]+}}

; VI-DAG: v_add_u16_e32 v{{[0-9]+}}, 0xfcb3, v{{[0-9]+}}
; VI-DAG: v_mov_b32_e32 v[[SCONST:[0-9]+]], 0xfffffc21
; VI-DAG: v_add_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v[[SCONST]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
define amdgpu_kernel void @v_test_add_v2i16_neg_constant(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %add = add <2 x i16> %a, <i16 -845, i16 -991>
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_add_v2i16_inline_neg1:
; GFX9PLUS: v_pk_sub_u16 v{{[0-9]+}}, v{{[0-9]+}}, 1 op_sel_hi:[1,0]{{$}}

; VI-DAG: v_mov_b32_e32 v[[SCONST:[0-9]+]], -1
; VI-DAG: flat_load_dword [[LOAD:v[0-9]+]]
; VI-DAG: v_add_u16_sdwa v{{[0-9]+}}, [[LOAD]], v[[SCONST]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_add_u16_e32 v{{[0-9]+}}, -1, [[LOAD]]
; VI: v_or_b32_e32
define amdgpu_kernel void @v_test_add_v2i16_inline_neg1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %add = add <2 x i16> %a, <i16 -1, i16 -1>
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_add_v2i16_inline_lo_zero_hi:
; GFX9PLUS: v_pk_add_u16 v{{[0-9]+}}, v{{[0-9]+}}, 32{{$}}

; VI: flat_load_dword
; VI-NOT: v_add_u16
; VI: v_and_b32_e32 v{{[0-9]+}}, 0xffff0000,
; VI: v_add_u16_e32 v{{[0-9]+}}, 32, v{{[0-9]+}}
; VI-NOT: v_add_u16
; VI: v_or_b32_e32
define amdgpu_kernel void @v_test_add_v2i16_inline_lo_zero_hi(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %add = add <2 x i16> %a, <i16 32, i16 0>
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; The high element gives fp
; GCN-LABEL: {{^}}v_test_add_v2i16_inline_fp_split:
; GFX9: s_mov_b32 [[K:s[0-9]+]], 1.0
; GFX9: v_pk_add_u16 v{{[0-9]+}}, v{{[0-9]+}}, [[K]]{{$}}
; GFX10: v_pk_add_u16 v{{[0-9]+}}, 0x3f80, v{{[0-9]+}}

; VI-NOT: v_add_u16
; VI: v_mov_b32_e32 v[[K:[0-9]+]], 0x3f80
; VI: v_add_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v[[K]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-NOT: v_add_u16
; VI: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
define amdgpu_kernel void @v_test_add_v2i16_inline_fp_split(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %add = add <2 x i16> %a, <i16 0, i16 16256>
  store <2 x i16> %add, <2 x i16> addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_v2i16_zext_to_v2i32:
; GFX9PLUS: global_load_dword [[A:v[0-9]+]]
; GFX9PLUS: global_load_dword [[B:v[0-9]+]]

; GFX9PLUS: v_pk_add_u16 [[ADD:v[0-9]+]], [[A]], [[B]]
; GFX9PLUS-DAG: v_and_b32_e32 v[[ELT0:[0-9]+]], 0xffff, [[ADD]]
; GFX9PLUS-DAG: v_lshrrev_b32_e32 v[[ELT1:[0-9]+]], 16, [[ADD]]
; GFX9PLUS: buffer_store_dwordx2 v[[[ELT0]]:[[ELT1]]]

; VI: flat_load_dword v[[A:[0-9]+]]
; VI: flat_load_dword v[[B:[0-9]+]]

; VI-NOT: and
; VI-NOT: shl
; VI: v_add_u16_e32 v[[ADD_LO:[0-9]+]], v[[A]], v[[B]]
; VI: v_add_u16_sdwa v[[ADD_HI:[0-9]+]], v[[A]], v[[B]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-NOT: and
; VI-NOT: shl
; VI: buffer_store_dwordx2 v[[[ADD_LO]]:[[ADD_HI]]]
define amdgpu_kernel void @v_test_add_v2i16_zext_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0, <2 x i16> addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i32>, <2 x i32> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in1, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %b = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in1
  %add = add <2 x i16> %a, %b
  %ext = zext <2 x i16> %add to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_v2i16_zext_to_v2i64:
; GFX9PLUS: v_mov_b32_e32 [[MASK:v[0-9+]]], 0xffff
; GFX9PLUS: global_load_dword [[A:v[0-9]+]]
; GFX9PLUS: global_load_dword [[B:v[0-9]+]]

; GFX9PLUS: v_pk_add_u16 [[ADD:v[0-9]+]], [[A]], [[B]]
; GFX9PLUS-DAG: v_and_b32_e32 v[[ELT0:[0-9]+]], 0xffff, [[ADD]]
; GFX9PLUS-DAG: v_and_b32_sdwa v{{[0-9]+}}, [[MASK]], v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; GFX9PLUS: buffer_store_dwordx4

; VI-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; VI-DAG: flat_load_dword v[[A:[0-9]+]]
; VI-DAG: flat_load_dword v[[B:[0-9]+]]

; VI-DAG: v_add_u16_e32
; VI: v_add_u16_sdwa v[[ADD_HI:[0-9]+]], v[[A]], v[[B]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1

; VI: buffer_store_dwordx4
define amdgpu_kernel void @v_test_add_v2i16_zext_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0, <2 x i16> addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i64>, <2 x i64> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in1, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %b = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in1
  %add = add <2 x i16> %a, %b
  %ext = zext <2 x i16> %add to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_v2i16_sext_to_v2i32:
; GFX9PLUS: global_load_dword [[A:v[0-9]+]]
; GFX9PLUS: global_load_dword [[B:v[0-9]+]]

; GFX9PLUS: v_pk_add_u16 [[ADD:v[0-9]+]], [[A]], [[B]]
; GFX9PLUS-DAG: v_bfe_i32 v[[ELT0:[0-9]+]], [[ADD]], 0, 16
; GFX9PLUS-DAG: v_ashrrev_i32_e32 v[[ELT1:[0-9]+]], 16, [[ADD]]
; GFX9PLUS: buffer_store_dwordx2 v[[[ELT0]]:[[ELT1]]]

; VI: v_add_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_add_u16_e32

; VI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; VI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; VI: buffer_store_dwordx2
define amdgpu_kernel void @v_test_add_v2i16_sext_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0, <2 x i16> addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i32>, <2 x i32> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in1, i32 %tid
  %a = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %b = load volatile <2 x i16>, <2 x i16> addrspace(1)* %gep.in1
  %add = add <2 x i16> %a, %b
  %ext = sext <2 x i16> %add to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; GCN-LABEL: {{^}}v_test_add_v2i16_sext_to_v2i64:
; GCN: {{flat|global}}_load_dword
; GCN: {{flat|global}}_load_dword

; GFX9PLUS: v_pk_add_u16
; GFX9PLUS: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}

; VI: v_add_u16_sdwa
; VI: v_add_u16_e32

; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
define amdgpu_kernel void @v_test_add_v2i16_sext_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i16> addrspace(1)* %in0, <2 x i16> addrspace(1)* %in1) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.out = getelementptr inbounds <2 x i64>, <2 x i64> addrspace(1)* %out, i32 %tid
  %gep.in0 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in0, i32 %tid
  %gep.in1 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in1, i32 %tid
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %gep.in0
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %gep.in1
  %add = add <2 x i16> %a, %b
  %ext = sext <2 x i16> %add to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
