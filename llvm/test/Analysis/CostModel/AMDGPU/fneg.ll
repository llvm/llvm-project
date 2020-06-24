; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; CHECK-LABEL: 'fneg_f32'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg float
define amdgpu_kernel void @fneg_f32(float addrspace(1)* %out, float addrspace(1)* %vaddr) {
  %vec = load float, float addrspace(1)* %vaddr
  %fadd = fadd float %vec, undef
  %fneg = fneg float %fadd
  store float %fneg, float addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_v2f32'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg <2 x float>
define amdgpu_kernel void @fneg_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr) {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %fadd = fadd <2 x float> %vec, undef
  %fneg = fneg <2 x float> %fadd
  store <2 x float> %fneg, <2 x float> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_v3f32'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg <3 x float>
define amdgpu_kernel void @fneg_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %vaddr) {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %vaddr
  %fadd = fadd <3 x float> %vec, undef
  %fneg = fneg <3 x float> %fadd
  store <3 x float> %fneg, <3 x float> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_v5f32'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg <5 x float>
define amdgpu_kernel void @fneg_v5f32(<5 x float> addrspace(1)* %out, <5 x float> addrspace(1)* %vaddr) {
  %vec = load <5 x float>, <5 x float> addrspace(1)* %vaddr
  %fadd = fadd <5 x float> %vec, undef
  %fneg = fneg <5 x float> %fadd
  store <5 x float> %fneg, <5 x float> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_f64'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg double
define amdgpu_kernel void @fneg_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr) {
  %vec = load double, double addrspace(1)* %vaddr
  %fadd = fadd double %vec, undef
  %fneg = fneg double %fadd
  store double %fneg, double addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_v2f64'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg <2 x double>
define amdgpu_kernel void @fneg_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %vaddr) {
  %vec = load <2 x double>, <2 x double> addrspace(1)* %vaddr
  %fadd = fadd <2 x double> %vec, undef
  %fneg = fneg <2 x double> %fadd
  store <2 x double> %fneg, <2 x double> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_v3f64'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg <3 x double>
define amdgpu_kernel void @fneg_v3f64(<3 x double> addrspace(1)* %out, <3 x double> addrspace(1)* %vaddr) {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %vaddr
  %fadd = fadd <3 x double> %vec, undef
  %fneg = fneg <3 x double> %fadd
  store <3 x double> %fneg, <3 x double> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_f16'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg half
define amdgpu_kernel void @fneg_f16(half addrspace(1)* %out, half addrspace(1)* %vaddr) {
  %vec = load half, half addrspace(1)* %vaddr
  %fadd = fadd half %vec, undef
  %fneg = fneg half %fadd
  store half %fneg, half addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_v2f16'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg <2 x half>
define amdgpu_kernel void @fneg_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr) {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %fadd = fadd <2 x half> %vec, undef
  %fneg = fneg <2 x half> %fadd
  store <2 x half> %fneg, <2 x half> addrspace(1)* %out
  ret void
}

; CHECK-LABEL: 'fneg_v3f16'
; CHECK: estimated cost of 0 for instruction:   %fneg = fneg <3 x half>
define amdgpu_kernel void @fneg_v3f16(<3 x half> addrspace(1)* %out, <3 x half> addrspace(1)* %vaddr) {
  %vec = load <3 x half>, <3 x half> addrspace(1)* %vaddr
  %fadd = fadd <3 x half> %vec, undef
  %fneg = fneg <3 x half> %fadd
  store <3 x half> %fneg, <3 x half> addrspace(1)* %out
  ret void
}
