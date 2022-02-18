; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,GFX89 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX89 %s

; DAGCombiner will transform:
; (fabs (f16 bitcast (i16 a))) => (f16 bitcast (and (i16 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; GCN-LABEL: {{^}}s_fabs_free_f16:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_and_b32 [[RESULT:s[0-9]+]], [[VAL]], 0x7fff
; GCN: v_mov_b32_e32 [[V_RESULT:v[0-9]+]], [[RESULT]]
; GCN: {{flat|global}}_store_short v{{.+}}, [[V_RESULT]]
define amdgpu_kernel void @s_fabs_free_f16(half addrspace(1)* %out, i16 %in) {
  %bc= bitcast i16 %in to half
  %fabs = call half @llvm.fabs.f16(half %bc)
  store half %fabs, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_fabs_f16:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_and_b32 [[RESULT:s[0-9]+]], [[VAL]], 0x7fff
; GCN: v_mov_b32_e32 [[V_RESULT:v[0-9]+]], [[RESULT]]
; GCN: {{flat|global}}_store_short v{{.+}}, [[V_RESULT]]
define amdgpu_kernel void @s_fabs_f16(half addrspace(1)* %out, half %in) {
  %fabs = call half @llvm.fabs.f16(half %in)
  store half %fabs, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_fabs_v2f16:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0x7fff7fff
define amdgpu_kernel void @s_fabs_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %in) {
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %in)
  store <2 x half> %fabs, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_fabs_v4f16:
; CI: s_load_dwordx2 s[[[LO:[0-9]+]]:[[HI:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, 0x2
; GFX89: s_load_dwordx2 s[[[LO:[0-9]+]]:[[HI:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, 0x8

; GCN: s_mov_b32 [[MASK:s[0-9]+]], 0x7fff7fff
; GCN-DAG: s_and_b32 s{{[0-9]+}}, s[[LO]], [[MASK]]
; GCN-DAG: s_and_b32 s{{[0-9]+}}, s[[HI]], [[MASK]]
; GCN: {{flat|global}}_store_dwordx2
define amdgpu_kernel void @s_fabs_v4f16(<4 x half> addrspace(1)* %out, <4 x half> %in) {
  %fabs = call <4 x half> @llvm.fabs.v4f16(<4 x half> %in)
  store <4 x half> %fabs, <4 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fabs_fold_f16:
; GCN: s_load_dword [[IN0:s[0-9]+]]
; GCN-DAG: s_lshr_b32 [[IN1:s[0-9]+]], [[IN0]], 16

; CI-DAG: v_cvt_f32_f16_e64 [[CVT0:v[0-9]+]], |[[IN0]]|
; CI-DAG: v_cvt_f32_f16_e32 [[ABS_CVT1:v[0-9]+]], [[IN1]]
; CI-DAG: v_mul_f32_e32 [[RESULT:v[0-9]+]], [[CVT0]], [[ABS_CVT1]]
; CI-DAG: v_cvt_f16_f32_e32 [[CVTRESULT:v[0-9]+]], [[RESULT]]
; CI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[CVTRESULT]]

; GFX89-NOT: and
; GFX89: v_mov_b32_e32 [[V_IN1:v[0-9]+]], [[IN1]]
; GFX89: v_mul_f16_e64 [[RESULT:v[0-9]+]], |[[IN0]]|, [[V_IN1]]
; GFX89: {{flat|global}}_store_short v{{.+}}, [[RESULT]]
define amdgpu_kernel void @fabs_fold_f16(half addrspace(1)* %out, half %in0, half %in1) {
  %fabs = call half @llvm.fabs.f16(half %in0)
  %fmul = fmul half %fabs, %in1
  store half %fmul, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fabs_v2f16:
; GCN: {{flat|global}}_load_dword [[VAL:v[0-9]+]]
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0x7fff7fff, [[VAL]]
define amdgpu_kernel void @v_fabs_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep.in, align 2
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  store <2 x half> %fabs, <2 x half> addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}fabs_free_v2f16:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_and_b32 s{{[0-9]+}}, [[VAL]], 0x7fff7fff
define amdgpu_kernel void @fabs_free_v2f16(<2 x half> addrspace(1)* %out, i32 %in) #0 {
  %bc = bitcast i32 %in to <2 x half>
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %bc)
  store <2 x half> %fabs, <2 x half> addrspace(1)* %out
  ret void
}

; FIXME: Should do fabs after conversion to avoid converting multiple
; times in this particular case.

; GCN-LABEL: {{^}}v_fabs_fold_self_v2f16:
; GCN: {{flat|global}}_load_dword [[VAL:v[0-9]+]]

; CI: v_lshrrev_b32_e32 [[VREG:v[0-9]+]], 16, v{{[0-9]+}}
; CI: v_cvt_f32_f16_e32 [[NORM:v[0-9]+]], [[VREG]]
; CI: v_cvt_f32_f16_e64 [[ABS:v[0-9]+]], {{\|}}[[VREG]]{{\|}}
; CI: v_mul_f32_e32 v{{[0-9]+}}, [[ABS]], [[NORM]]
; CI: v_cvt_f16_f32
; CI: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CI: v_cvt_f16_f32

; VI: v_mul_f16_sdwa v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_mul_f16_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}}

; GFX9: v_and_b32_e32 [[FABS:v[0-9]+]], 0x7fff7fff, [[VAL]]
; GFX9: v_pk_mul_f16 v{{[0-9]+}}, [[FABS]], v{{[0-9]+$}}
define amdgpu_kernel void @v_fabs_fold_self_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %in, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %fmul = fmul <2 x half> %fabs, %val
  store <2 x half> %fmul, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fabs_fold_v2f16:
; GCN: {{flat|global}}_load_dword [[VAL:v[0-9]+]]

; CI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; CI: v_cvt_f32_f16_e32
; CI: v_cvt_f32_f16_e32
; CI: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; CI: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CI: v_cvt_f16_f32
; CI: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CI: v_cvt_f16_f32

; VI: v_mul_f16_sdwa v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI: v_mul_f16_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, s{{[0-9]+}}

; GFX9: v_and_b32_e32 [[FABS:v[0-9]+]], 0x7fff7fff, [[VAL]]
; GFX9: v_pk_mul_f16 v{{[0-9]+}}, [[FABS]], s{{[0-9]+$}}
define amdgpu_kernel void @v_fabs_fold_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in, i32 %other.val) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %in, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %other.val.cvt = bitcast i32 %other.val to <2 x half>
  %fmul = fmul <2 x half> %fabs, %other.val.cvt
  store <2 x half> %fmul, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_extract_fabs_fold_v2f16:
; GCN-DAG: {{flat|global}}_load_dword [[VAL:v[0-9]+]]
; CI-DAG: v_mul_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
; CI-DAG: v_add_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}

; GFX89-DAG: v_mul_f16_e64 v{{[0-9]+}}, |[[VAL]]|, 4.0
; GFX89-DAG: v_mov_b32_e32 [[CONST2:v[0-9]+]], 0x4000
; GFX89-DAG: v_add_f16_sdwa v{{[0-9]+}}, |[[VAL]]|, [[CONST2]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
define amdgpu_kernel void @v_extract_fabs_fold_v2f16(<2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep.in
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %elt0 = extractelement <2 x half> %fabs, i32 0
  %elt1 = extractelement <2 x half> %fabs, i32 1

  %fmul0 = fmul half %elt0, 4.0
  %fadd1 = fadd half %elt1, 2.0
  store volatile half %fmul0, half addrspace(1)* undef
  store volatile half %fadd1, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_extract_fabs_no_fold_v2f16:
; GCN: {{flat|global}}_load_dword [[VAL:v[0-9]+]]
; GCN: v_and_b32_e32 [[AND:v[0-9]+]], 0x7fff7fff, [[VAL]]


; VI: v_bfe_u32 v{{[0-9]+}}, v{{[0-9]+}}, 16, 15
; VI: flat_store_short

; GFX9: global_store_short_d16_hi v{{\[[0-9]+:[0-9]+\]}}, [[AND]], off
define amdgpu_kernel void @v_extract_fabs_no_fold_v2f16(<2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep.in
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %elt0 = extractelement <2 x half> %fabs, i32 0
  %elt1 = extractelement <2 x half> %fabs, i32 1
  store volatile half %elt0, half addrspace(1)* undef
  store volatile half %elt1, half addrspace(1)* undef
  ret void
}

declare half @llvm.fabs.f16(half) #1
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #1
declare <4 x half> @llvm.fabs.v4f16(<4 x half>) #1
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
