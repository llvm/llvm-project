; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s

declare double @llvm.ceil.f64(double) nounwind readnone
declare <2 x double> @llvm.ceil.v2f64(<2 x double>) nounwind readnone
declare <3 x double> @llvm.ceil.v3f64(<3 x double>) nounwind readnone
declare <4 x double> @llvm.ceil.v4f64(<4 x double>) nounwind readnone
declare <8 x double> @llvm.ceil.v8f64(<8 x double>) nounwind readnone
declare <16 x double> @llvm.ceil.v16f64(<16 x double>) nounwind readnone

; FUNC-LABEL: {{^}}fceil_f64:
; CI: v_ceil_f64_e32
; SI: s_bfe_u32 [[SEXP:s[0-9]+]], {{s[0-9]+}}, 0xb0014
; SI-DAG: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80000000
; FIXME: We should be using s_addk_i32 here, but the reg allocation hints
;        are not always followed.
; SI-DAG: s_add_i32 [[SEXP0:s[0-9]+]], [[SEXP]], 0xfffffc01
; SI-DAG: s_lshr_b64 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], [[SEXP0]]
; SI-DAG: s_not_b64
; SI-DAG: s_and_b64
; SI-DAG: cmp_gt_i32
; SI-DAG: cndmask_b32
; SI-DAG: cndmask_b32
; SI-DAG: cmp_lt_i32
; SI-DAG: cndmask_b32
; SI-DAG: cndmask_b32
; SI-DAG: v_cmp_gt_f64
; SI-DAG: v_cmp_lg_f64
; SI-DAG: v_cndmask_b32
; SI: v_cndmask_b32
; SI: v_add_f64
; SI: s_endpgm
define void @fceil_f64(double addrspace(1)* %out, double %x) {
  %y = call double @llvm.ceil.f64(double %x) nounwind readnone
  store double %y, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fceil_v2f64:
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
define void @fceil_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %x) {
  %y = call <2 x double> @llvm.ceil.v2f64(<2 x double> %x) nounwind readnone
  store <2 x double> %y, <2 x double> addrspace(1)* %out
  ret void
}

; FIXME-FUNC-LABEL: {{^}}fceil_v3f64:
; FIXME-CI: v_ceil_f64_e32
; FIXME-CI: v_ceil_f64_e32
; FIXME-CI: v_ceil_f64_e32
; define void @fceil_v3f64(<3 x double> addrspace(1)* %out, <3 x double> %x) {
;   %y = call <3 x double> @llvm.ceil.v3f64(<3 x double> %x) nounwind readnone
;   store <3 x double> %y, <3 x double> addrspace(1)* %out
;   ret void
; }

; FUNC-LABEL: {{^}}fceil_v4f64:
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
define void @fceil_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %x) {
  %y = call <4 x double> @llvm.ceil.v4f64(<4 x double> %x) nounwind readnone
  store <4 x double> %y, <4 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fceil_v8f64:
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
define void @fceil_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %x) {
  %y = call <8 x double> @llvm.ceil.v8f64(<8 x double> %x) nounwind readnone
  store <8 x double> %y, <8 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fceil_v16f64:
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
; CI: v_ceil_f64_e32
define void @fceil_v16f64(<16 x double> addrspace(1)* %out, <16 x double> %x) {
  %y = call <16 x double> @llvm.ceil.v16f64(<16 x double> %x) nounwind readnone
  store <16 x double> %y, <16 x double> addrspace(1)* %out
  ret void
}
