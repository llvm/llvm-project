; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=bonaire < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s

declare double @llvm.trunc.f64(double) nounwind readnone
declare <2 x double> @llvm.trunc.v2f64(<2 x double>) nounwind readnone
declare <3 x double> @llvm.trunc.v3f64(<3 x double>) nounwind readnone
declare <4 x double> @llvm.trunc.v4f64(<4 x double>) nounwind readnone
declare <8 x double> @llvm.trunc.v8f64(<8 x double>) nounwind readnone
declare <16 x double> @llvm.trunc.v16f64(<16 x double>) nounwind readnone

; FUNC-LABEL: {{^}}v_ftrunc_f64:
; CI: v_trunc_f64
; SI: s_bfe_u32 {{s[0-9]+}}, {{s[0-9]+}}, 0xb0014
; SI: s_endpgm
define amdgpu_kernel void @v_ftrunc_f64(double addrspace(1)* %out, double addrspace(1)* %in) {
  %x = load double, double addrspace(1)* %in, align 8
  %y = call double @llvm.trunc.f64(double %x) nounwind readnone
  store double %y, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_f64:
; CI: v_trunc_f64_e32

; SI: s_bfe_u32 [[SEXP:s[0-9]+]], {{s[0-9]+}}, 0xb0014
; SI-DAG: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80000000
; SI-DAG: s_add_i32 [[SEXP1:s[0-9]+]], [[SEXP]], 0xfffffc01
; SI-DAG: s_lshr_b64 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], [[SEXP1]]
; SI-DAG: s_andn2_b64
; SI-DAG: cmp_gt_i32
; SI-DAG: s_cselect_b32
; SI-DAG: s_cselect_b32
; SI-DAG: cmp_lt_i32
; SI-DAG: s_cselect_b32
; SI-DAG: s_cselect_b32
; SI: s_endpgm
define amdgpu_kernel void @ftrunc_f64(double addrspace(1)* %out, double %x) {
  %y = call double @llvm.trunc.f64(double %x) nounwind readnone
  store double %y, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v2f64:
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
define amdgpu_kernel void @ftrunc_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %x) {
  %y = call <2 x double> @llvm.trunc.v2f64(<2 x double> %x) nounwind readnone
  store <2 x double> %y, <2 x double> addrspace(1)* %out
  ret void
}

; FIXME-FUNC-LABEL: {{^}}ftrunc_v3f64:
; FIXME-CI: v_trunc_f64_e32
; FIXME-CI: v_trunc_f64_e32
; FIXME-CI: v_trunc_f64_e32
; define amdgpu_kernel void @ftrunc_v3f64(<3 x double> addrspace(1)* %out, <3 x double> %x) {
;   %y = call <3 x double> @llvm.trunc.v3f64(<3 x double> %x) nounwind readnone
;   store <3 x double> %y, <3 x double> addrspace(1)* %out
;   ret void
; }

; FUNC-LABEL: {{^}}ftrunc_v4f64:
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
define amdgpu_kernel void @ftrunc_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %x) {
  %y = call <4 x double> @llvm.trunc.v4f64(<4 x double> %x) nounwind readnone
  store <4 x double> %y, <4 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v8f64:
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
define amdgpu_kernel void @ftrunc_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %x) {
  %y = call <8 x double> @llvm.trunc.v8f64(<8 x double> %x) nounwind readnone
  store <8 x double> %y, <8 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ftrunc_v16f64:
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
; CI: v_trunc_f64_e32
define amdgpu_kernel void @ftrunc_v16f64(<16 x double> addrspace(1)* %out, <16 x double> %x) {
  %y = call <16 x double> @llvm.trunc.v16f64(<16 x double> %x) nounwind readnone
  store <16 x double> %y, <16 x double> addrspace(1)* %out
  ret void
}
