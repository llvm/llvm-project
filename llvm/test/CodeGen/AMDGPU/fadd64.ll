; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}v_fadd_f64:
; CHECK: v_add_f64 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @v_fadd_f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                        ptr addrspace(1) %in2) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep1 = getelementptr inbounds double, ptr addrspace(1) %in1, i32 %tid
  %gep2 = getelementptr inbounds double, ptr addrspace(1) %in2, i32 %tid
  %r0 = load double, ptr addrspace(1) %gep1
  %r1 = load double, ptr addrspace(1) %gep2
  %r2 = fadd double %r0, %r1
  store double %r2, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}s_fadd_f64:
; CHECK: v_add_f64 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @s_fadd_f64(ptr addrspace(1) %out, double %r0, double %r1) {
  %r2 = fadd double %r0, %r1
  store double %r2, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}v_fadd_v2f64:
; CHECK: v_add_f64
; CHECK: v_add_f64
; CHECK: _store_dwordx4
define amdgpu_kernel void @v_fadd_v2f64(ptr addrspace(1) %out, ptr addrspace(1) %in1,
                          ptr addrspace(1) %in2) {
  %r0 = load <2 x double>, ptr addrspace(1) %in1
  %r1 = load <2 x double>, ptr addrspace(1) %in2
  %r2 = fadd <2 x double> %r0, %r1
  store <2 x double> %r2, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}s_fadd_v2f64:
; CHECK: v_add_f64 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
; CHECK: v_add_f64 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
; CHECK: _store_dwordx4
define amdgpu_kernel void @s_fadd_v2f64(ptr addrspace(1) %out, <2 x double> %r0, <2 x double> %r1) {
  %r2 = fadd <2 x double> %r0, %r1
  store <2 x double> %r2, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
