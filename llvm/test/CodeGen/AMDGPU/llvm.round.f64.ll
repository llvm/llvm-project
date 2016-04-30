; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}round_f64:
; SI: s_endpgm
define void @round_f64(double addrspace(1)* %out, double %x) #0 {
  %result = call double @llvm.round.f64(double %x) #1
  store double %result, double addrspace(1)* %out
  ret void
}

; This is a pretty large function, so just test a few of the
; instructions that are necessary.

; FUNC-LABEL: {{^}}v_round_f64:
; SI: buffer_load_dwordx2
; SI-DAG: v_bfe_u32 [[EXP:v[0-9]+]], v{{[0-9]+}}, 20, 11

; SI-DAG: v_not_b32_e32
; SI-DAG: v_not_b32_e32

; SI-DAG: v_cmp_eq_i32

; SI-DAG: s_mov_b32 [[BFIMASK:s[0-9]+]], 0x7fffffff
; SI-DAG: v_cmp_gt_i32
; SI-DAG: v_bfi_b32 [[COPYSIGN:v[0-9]+]], [[BFIMASK]]

; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @v_round_f64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep = getelementptr double, double addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr double, double addrspace(1)* %out, i32 %tid
  %x = load double, double addrspace(1)* %gep
  %result = call double @llvm.round.f64(double %x) #1
  store double %result, double addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}round_v2f64:
; SI: s_endpgm
define void @round_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %in) #0 {
  %result = call <2 x double> @llvm.round.v2f64(<2 x double> %in) #1
  store <2 x double> %result, <2 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}round_v4f64:
; SI: s_endpgm
define void @round_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %in) #0 {
  %result = call <4 x double> @llvm.round.v4f64(<4 x double> %in) #1
  store <4 x double> %result, <4 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}round_v8f64:
; SI: s_endpgm
define void @round_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %in) #0 {
  %result = call <8 x double> @llvm.round.v8f64(<8 x double> %in) #1
  store <8 x double> %result, <8 x double> addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

declare double @llvm.round.f64(double) #1
declare <2 x double> @llvm.round.v2f64(<2 x double>) #1
declare <4 x double> @llvm.round.v4f64(<4 x double>) #1
declare <8 x double> @llvm.round.v8f64(<8 x double>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
