; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK: {{^}}fptrunc:
; CHECK: v_cvt_f32_f64_e32
define void @fptrunc(float addrspace(1)* %out, double %in) {
  %result = fptrunc double %in to float
  store float %result, float addrspace(1)* %out
  ret void
}
