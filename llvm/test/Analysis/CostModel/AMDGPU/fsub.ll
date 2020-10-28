; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FASTF64,FASTF16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOWF64,SLOWF16,ALL %s
; RUN: opt -cost-model -analyze -cost-kind=code-size -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=SIZEALL,FASTF16,ALL %s
; RUN: opt -cost-model -analyze -cost-kind=code-size -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SIZEALL,SLOWF16,ALL %s

; ALL-LABEL: 'fsub_f32'
; ALL: estimated cost of 1 for {{.*}} fsub float
define amdgpu_kernel void @fsub_f32(float addrspace(1)* %out, float addrspace(1)* %vaddr, float %b) #0 {
  %vec = load float, float addrspace(1)* %vaddr
  %add = fsub float %vec, %b
  store float %add, float addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_v2f32'
; ALL: estimated cost of 2 for {{.*}} fsub <2 x float>
define amdgpu_kernel void @fsub_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr, <2 x float> %b) #0 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %add = fsub <2 x float> %vec, %b
  store <2 x float> %add, <2 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_v3f32'
; ALL: estimated cost of 3 for {{.*}} fsub <3 x float>
define amdgpu_kernel void @fsub_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %vaddr, <3 x float> %b) #0 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %vaddr
  %add = fsub <3 x float> %vec, %b
  store <3 x float> %add, <3 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_v5f32'
; ALL: estimated cost of 5 for {{.*}} fsub <5 x float>
define amdgpu_kernel void @fsub_v5f32(<5 x float> addrspace(1)* %out, <5 x float> addrspace(1)* %vaddr, <5 x float> %b) #0 {
  %vec = load <5 x float>, <5 x float> addrspace(1)* %vaddr
  %add = fsub <5 x float> %vec, %b
  store <5 x float> %add, <5 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_f64'
; FASTF64: estimated cost of 2 for {{.*}} fsub double
; SLOWF64: estimated cost of 4 for {{.*}} fsub double
; SIZEALL: estimated cost of 2 for {{.*}} fsub double
define amdgpu_kernel void @fsub_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr, double %b) #0 {
  %vec = load double, double addrspace(1)* %vaddr
  %add = fsub double %vec, %b
  store double %add, double addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_v2f64'
; FASTF64: estimated cost of 4 for {{.*}} fsub <2 x double>
; SLOWF64: estimated cost of 8 for {{.*}} fsub <2 x double>
; SIZEALL: estimated cost of 4 for {{.*}} fsub <2 x double>
define amdgpu_kernel void @fsub_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %vaddr, <2 x double> %b) #0 {
  %vec = load <2 x double>, <2 x double> addrspace(1)* %vaddr
  %add = fsub <2 x double> %vec, %b
  store <2 x double> %add, <2 x double> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_v3f64'
; FASTF64: estimated cost of 6 for {{.*}} fsub <3 x double>
; SLOWF64: estimated cost of 12 for {{.*}} fsub <3 x double>
; SIZEALL: estimated cost of 6 for {{.*}} fsub <3 x double>
define amdgpu_kernel void @fsub_v3f64(<3 x double> addrspace(1)* %out, <3 x double> addrspace(1)* %vaddr, <3 x double> %b) #0 {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %vaddr
  %add = fsub <3 x double> %vec, %b
  store <3 x double> %add, <3 x double> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_f16'
; ALL: estimated cost of 1 for {{.*}} fsub half
define amdgpu_kernel void @fsub_f16(half addrspace(1)* %out, half addrspace(1)* %vaddr, half %b) #0 {
  %vec = load half, half addrspace(1)* %vaddr
  %add = fsub half %vec, %b
  store half %add, half addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_v2f16'
; SLOWF16: estimated cost of 2 for {{.*}} fsub <2 x half>
; FASTF16: estimated cost of 1 for {{.*}} fsub <2 x half>
define amdgpu_kernel void @fsub_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr, <2 x half> %b) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %add = fsub <2 x half> %vec, %b
  store <2 x half> %add, <2 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_v3f16'
; SLOWF16: estimated cost of 4 for {{.*}} fsub <3 x half>
; FASTF16: estimated cost of 2 for {{.*}} fsub <3 x half>
define amdgpu_kernel void @fsub_v3f16(<3 x half> addrspace(1)* %out, <3 x half> addrspace(1)* %vaddr, <3 x half> %b) #0 {
  %vec = load <3 x half>, <3 x half> addrspace(1)* %vaddr
  %add = fsub <3 x half> %vec, %b
  store <3 x half> %add, <3 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fsub_v4f16'
; SLOWF16: estimated cost of 4 for {{.*}} fsub <4 x half>
; FASTF16: estimated cost of 2 for {{.*}} fsub <4 x half>
define amdgpu_kernel void @fsub_v4f16(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %vaddr, <4 x half> %b) #0 {
  %vec = load <4 x half>, <4 x half> addrspace(1)* %vaddr
  %add = fsub <4 x half> %vec, %b
  store <4 x half> %add, <4 x half> addrspace(1)* %out
  ret void
}
