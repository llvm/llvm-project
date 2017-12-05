; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx901 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GFX9 %s

; DAGCombiner will transform:
; (fabs (f16 bitcast (i16 a))) => (f16 bitcast (and (i16 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; GCN-LABEL: {{^}}s_fabs_free_f16:
; GCN: {{flat|global}}_load_ushort [[VAL:v[0-9]+]],
; GCN: v_and_b32_e32 [[RESULT:v[0-9]+]], 0x7fff, [[VAL]]
; GCN: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

define amdgpu_kernel void @s_fabs_free_f16(half addrspace(1)* %out, i16 %in) {
  %bc= bitcast i16 %in to half
  %fabs = call half @llvm.fabs.f16(half %bc)
  store half %fabs, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_fabs_f16:
; CI: flat_load_ushort [[VAL:v[0-9]+]],
; CI: v_and_b32_e32 [[RESULT:v[0-9]+]], 0x7fff, [[VAL]]
; CI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @s_fabs_f16(half addrspace(1)* %out, half %in) {
  %fabs = call half @llvm.fabs.f16(half %in)
  store half %fabs, half addrspace(1)* %out
  ret void
}

; FIXME: Should be able to use single and
; GCN-LABEL: {{^}}s_fabs_v2f16:
; CI: s_movk_i32 [[MASK:s[0-9]+]], 0x7fff
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]]
; CI: v_lshlrev_b32_e32 v{{[0-9]+}}, 16,
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]]
; CI: v_or_b32_e32

; VI: flat_load_ushort [[HI:v[0-9]+]]
; VI: flat_load_ushort [[LO:v[0-9]+]]
; VI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7fff{{$}}
; VI-DAG: v_and_b32_e32 [[FABS_LO:v[0-9]+]], [[HI]], [[MASK]]
; VI-DAG: v_and_b32_sdwa [[FABS_HI:v[0-9]+]], [[LO]], [[MASK]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, [[FABS_LO]], [[FABS_HI]]
; VI: flat_store_dword

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: s_and_b32 s{{[0-9]+}}, [[VAL]], 0x7fff7fff
define amdgpu_kernel void @s_fabs_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %in) {
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %in)
  store <2 x half> %fabs, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_fabs_v4f16:
; CI: s_movk_i32 [[MASK:s[0-9]+]], 0x7fff
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]]
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]]
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]]
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]]

; VI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7fff{{$}}
; VI-DAG: v_and_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, [[MASK]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_and_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, [[MASK]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_and_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[MASK]]
; VI-DAG: v_and_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[MASK]]
; VI-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI:     v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; GCN: {{flat|global}}_store_dwordx2
define amdgpu_kernel void @s_fabs_v4f16(<4 x half> addrspace(1)* %out, <4 x half> %in) {
  %fabs = call <4 x half> @llvm.fabs.v4f16(<4 x half> %in)
  store <4 x half> %fabs, <4 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fabs_fold_f16:
; GCN: {{flat|global}}_load_ushort [[IN0:v[0-9]+]]
; GCN: {{flat|global}}_load_ushort [[IN1:v[0-9]+]]

; CI-DAG: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], [[IN0]]
; CI-DAG: v_cvt_f32_f16_e64 [[ABS_CVT1:v[0-9]+]], |[[IN1]]|
; CI: v_mul_f32_e32 [[RESULT:v[0-9]+]], [[ABS_CVT1]], [[CVT0]]
; CI: v_cvt_f16_f32_e32 [[CVTRESULT:v[0-9]+]], [[RESULT]]
; CI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[CVTRESULT]]

; VI-NOT: and
; VI: v_mul_f16_e64 [[RESULT:v[0-9]+]], |[[IN1]]|, [[IN0]]
; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
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

; GCN-LABEL: {{^}}v_fabs_fold_v2f16:
; GCN: {{flat|global}}_load_dword [[VAL:v[0-9]+]]

; CI: v_cvt_f32_f16_e32
; CI: v_cvt_f32_f16_e32
; CI: v_mul_f32_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}}
; CI: v_cvt_f16_f32
; CI: v_mul_f32_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}}
; CI: v_cvt_f16_f32

; VI: v_mul_f16_sdwa v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_mul_f16_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}}

; GFX9: v_and_b32_e32 [[FABS:v[0-9]+]], 0x7fff7fff, [[VAL]]
; GFX9: v_pk_mul_f16 v{{[0-9]+}}, [[FABS]], v{{[0-9]+$}}
define amdgpu_kernel void @v_fabs_fold_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %in, i32 %tid
  %val = load <2 x half>, <2 x half> addrspace(1)* %gep
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %fmul = fmul <2 x half> %fabs, %val
  store <2 x half> %fmul, <2 x half> addrspace(1)* %out
  ret void
}

declare half @llvm.fabs.f16(half) #1
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #1
declare <4 x half> @llvm.fabs.v4f16(<4 x half>) #1
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
