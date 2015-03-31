; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK: {{^}}fpext:
; CHECK: v_cvt_f64_f32_e32
define void @fpext(double addrspace(1)* %out, float %in) {
  %result = fpext float %in to double
  store double %result, double addrspace(1)* %out
  ret void
}
