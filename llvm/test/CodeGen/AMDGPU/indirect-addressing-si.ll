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

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #1 = { nounwind readnone }
