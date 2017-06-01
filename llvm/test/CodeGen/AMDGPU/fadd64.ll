; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}v_fadd_f64:
; CHECK: v_add_f64 {{v[[0-9]+:[0-9]+]}}, {{v[[0-9]+:[0-9]+]}}, {{v[[0-9]+:[0-9]+]}}
define void @v_fadd_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                        double addrspace(1)* %in2) {
  %r0 = load double, double addrspace(1)* %in1
  %r1 = load double, double addrspace(1)* %in2
  %r2 = fadd double %r0, %r1
  store double %r2, double addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}s_fadd_f64:
; CHECK: v_add_f64 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define void @s_fadd_f64(double addrspace(1)* %out, double %r0, double %r1) {
  %r2 = fadd double %r0, %r1
  store double %r2, double addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}v_fadd_v2f64:
; CHECK: v_add_f64
; CHECK: v_add_f64
; CHECK: _store_dwordx4
define void @v_fadd_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %in1,
                          <2 x double> addrspace(1)* %in2) {
  %r0 = load <2 x double>, <2 x double> addrspace(1)* %in1
  %r1 = load <2 x double>, <2 x double> addrspace(1)* %in2
  %r2 = fadd <2 x double> %r0, %r1
  store <2 x double> %r2, <2 x double> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}s_fadd_v2f64:
; CHECK: v_add_f64 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
; CHECK: v_add_f64 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
; CHECK: _store_dwordx4
define void @s_fadd_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %r0, <2 x double> %r1) {
  %r2 = fadd <2 x double> %r0, %r1
  store <2 x double> %r2, <2 x double> addrspace(1)* %out
  ret void
}
