; RUN: llc -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; DAGCombiner will transform:
; (fabs (f16 bitcast (i16 a))) => (f16 bitcast (and (i16 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; GCN-LABEL: {{^}}fabs_free_f16:
; GCN: flat_load_ushort [[VAL:v[0-9]+]],
; GCN: v_and_b32_e32 [[RESULT:v[0-9]+]], 0x7fff, [[VAL]]
; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]

define void @fabs_free_f16(half addrspace(1)* %out, i16 %in) {
  %bc= bitcast i16 %in to half
  %fabs = call half @llvm.fabs.f16(half %bc)
  store half %fabs, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fabs_f16:
; CI: flat_load_ushort [[VAL:v[0-9]+]],
; CI: v_and_b32_e32 [[CVT0:v[0-9]+]], 0x7fff, [[VAL]]
; CI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define void @fabs_f16(half addrspace(1)* %out, half %in) {
  %fabs = call half @llvm.fabs.f16(half %in)
  store half %fabs, half addrspace(1)* %out
  ret void
}

; FIXME: Should be able to use single and
; GCN-LABEL: {{^}}fabs_v2f16:

; CI: s_movk_i32 [[MASK:s[0-9]+]], 0x7fff
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}

; VI: flat_load_ushort [[LO:v[0-9]+]]
; VI: flat_load_ushort [[HI:v[0-9]+]]
; VI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7fff{{$}}
; VI-DAG: v_and_b32_e32 [[FABS_LO:v[0-9]+]], [[MASK]], [[LO]]
; VI-DAG: v_and_b32_e32 [[FABS_LO:v[0-9]+]], [[MASK]], [[HI]]
; VI-DAG: v_lshlrev_b32_e32 v{{[0-9]+}}, 16,
; VI-DAG: v_and_b32_e32 v{{[0-9]+}}, 0xffff,
; VI: v_or_b32
; VI: flat_store_dword
define void @fabs_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %in) {
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %in)
  store <2 x half> %fabs, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fabs_v4f16:
; CI: s_movk_i32 [[MASK:s[0-9]+]], 0x7fff
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}

; VI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x7fff{{$}}
; VI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}
; VI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}
; VI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}
; VI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}

; GCN: flat_store_dwordx2
define void @fabs_v4f16(<4 x half> addrspace(1)* %out, <4 x half> %in) {
  %fabs = call <4 x half> @llvm.fabs.v4f16(<4 x half> %in)
  store <4 x half> %fabs, <4 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fabs_fold_f16:
; GCN: flat_load_ushort [[IN0:v[0-9]+]]
; GCN: flat_load_ushort [[IN1:v[0-9]+]]

; CI-DAG: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], [[IN0]]
; CI-DAG: v_cvt_f32_f16_e64 [[ABS_CVT1:v[0-9]+]], |[[IN1]]|
; CI: v_mul_f32_e32 [[RESULT:v[0-9]+]],  [[CVT0]], [[ABS_CVT1]]
; CI: v_cvt_f16_f32_e32 [[CVTRESULT:v[0-9]+]], [[RESULT]]
; CI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[CVTRESULT]]

; VI-NOT: and
; VI: v_mul_f16_e64 [[RESULT:v[0-9]+]], |[[IN1]]|, [[IN0]]
; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define void @fabs_fold_f16(half addrspace(1)* %out, half %in0, half %in1) {
  %fabs = call half @llvm.fabs.f16(half %in0)
  %fmul = fmul half %fabs, %in1
  store half %fmul, half addrspace(1)* %out
  ret void
}

declare half @llvm.fabs.f16(half) readnone
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) readnone
declare <4 x half> @llvm.fabs.v4f16(<4 x half>) readnone
