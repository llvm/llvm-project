; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck -check-prefix=SI %s
; RUN: llc -mtriple=amdgpu8.02 < %s | FileCheck -check-prefix=SI %s

declare double @llvm.amdgcn.trig.preop.f64(double, i32) nounwind readnone

; SI-LABEL: {{^}}test_trig_preop_f64:
; SI: v_trig_preop_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], v[0:1], v2
; SI: s_setpc_b64
define double @test_trig_preop_f64(double %a, i32 %b) nounwind {
  %result = call double @llvm.amdgcn.trig.preop.f64(double %a, i32 %b) nounwind readnone
  ret double %result
}

; SI-LABEL: {{^}}test_trig_preop_f64_imm_segment:
; SI: v_trig_preop_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], v[0:1], 7
; SI: s_setpc_b64
define double @test_trig_preop_f64_imm_segment(double %a) nounwind {
  %result = call double @llvm.amdgcn.trig.preop.f64(double %a, i32 7) nounwind readnone
  ret double %result
}
