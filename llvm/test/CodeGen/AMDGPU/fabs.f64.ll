; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

declare double @fabs(double) readnone
declare double @llvm.fabs.f64(double) readnone
declare <2 x double> @llvm.fabs.v2f64(<2 x double>) readnone
declare <4 x double> @llvm.fabs.v4f64(<4 x double>) readnone

; FUNC-LABEL: {{^}}v_fabs_f64:
; SI: v_and_b32
; SI: s_endpgm
define amdgpu_kernel void @v_fabs_f64(double addrspace(1)* %out, double addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %tidext = sext i32 %tid to i64
  %gep = getelementptr double, double addrspace(1)* %in, i64 %tidext
  %val = load double, double addrspace(1)* %gep, align 8
  %fabs = call double @llvm.fabs.f64(double %val)
  store double %fabs, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_f64:
; SI: s_bitset0_b32
; SI: s_endpgm
define amdgpu_kernel void @fabs_f64(double addrspace(1)* %out, double %in) {
  %fabs = call double @llvm.fabs.f64(double %in)
  store double %fabs, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_v2f64:
; SI: s_bitset0_b32
; SI: s_bitset0_b32
; SI: s_endpgm
define amdgpu_kernel void @fabs_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %in) {
  %fabs = call <2 x double> @llvm.fabs.v2f64(<2 x double> %in)
  store <2 x double> %fabs, <2 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_v4f64:
; SI: s_bitset0_b32
; SI: s_bitset0_b32
; SI: s_bitset0_b32
; SI: s_bitset0_b32
; SI: s_endpgm
define amdgpu_kernel void @fabs_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %in) {
  %fabs = call <4 x double> @llvm.fabs.v4f64(<4 x double> %in)
  store <4 x double> %fabs, <4 x double> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}fabs_fold_f64:
; SI: s_load_dwordx2 [[ABS_VALUE:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x13
; SI-NOT: and
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\]}}, |[[ABS_VALUE]]|, {{v\[[0-9]+:[0-9]+\]}}
; SI: s_endpgm
define amdgpu_kernel void @fabs_fold_f64(double addrspace(1)* %out, [8 x i32], double %in0, [8 x i32], double %in1) {
  %fabs = call double @llvm.fabs.f64(double %in0)
  %fmul = fmul double %fabs, %in1
  store double %fmul, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}fabs_fn_fold_f64:
; SI: s_load_dwordx2 [[ABS_VALUE:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x13
; SI-NOT: and
; SI: v_mul_f64 {{v\[[0-9]+:[0-9]+\]}}, |[[ABS_VALUE]]|, {{v\[[0-9]+:[0-9]+\]}}
; SI: s_endpgm
define amdgpu_kernel void @fabs_fn_fold_f64(double addrspace(1)* %out, [8 x i32], double %in0, [8 x i32], double %in1) {
  %fabs = call double @fabs(double %in0)
  %fmul = fmul double %fabs, %in1
  store double %fmul, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_free_f64:
; SI: s_bitset0_b32
; SI: s_endpgm
define amdgpu_kernel void @fabs_free_f64(double addrspace(1)* %out, i64 %in) {
  %bc= bitcast i64 %in to double
  %fabs = call double @llvm.fabs.f64(double %bc)
  store double %fabs, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_fn_free_f64:
; SI: s_bitset0_b32
; SI: s_endpgm
define amdgpu_kernel void @fabs_fn_free_f64(double addrspace(1)* %out, i64 %in) {
  %bc= bitcast i64 %in to double
  %fabs = call double @fabs(double %bc)
  store double %fabs, double addrspace(1)* %out
  ret void
}
