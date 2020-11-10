; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900  -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FASTF64,FASTF16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOWF64,SLOWF16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900  -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FASTF16,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOWF16,SIZEALL,ALL %s

; ALL-LABEL: 'fadd_f32'
; ALL: estimated cost of 1 for {{.*}} fadd float
define amdgpu_kernel void @fadd_f32(float addrspace(1)* %out, float addrspace(1)* %vaddr, float %b) #0 {
  %vec = load float, float addrspace(1)* %vaddr
  %add = fadd float %vec, %b
  store float %add, float addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_v2f32'
; ALL: estimated cost of 2 for {{.*}} fadd <2 x float>
define amdgpu_kernel void @fadd_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr, <2 x float> %b) #0 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %add = fadd <2 x float> %vec, %b
  store <2 x float> %add, <2 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_v3f32'
; ALL: estimated cost of 3 for {{.*}} fadd <3 x float>
define amdgpu_kernel void @fadd_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %vaddr, <3 x float> %b) #0 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %vaddr
  %add = fadd <3 x float> %vec, %b
  store <3 x float> %add, <3 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_v5f32'
; ALL: estimated cost of 5 for {{.*}} fadd <5 x float>
define amdgpu_kernel void @fadd_v5f32(<5 x float> addrspace(1)* %out, <5 x float> addrspace(1)* %vaddr, <5 x float> %b) #0 {
  %vec = load <5 x float>, <5 x float> addrspace(1)* %vaddr
  %add = fadd <5 x float> %vec, %b
  store <5 x float> %add, <5 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_f64'
; FASTF64: estimated cost of 2 for {{.*}} fadd double
; SLOWF64: estimated cost of 4 for {{.*}} fadd double
; SIZEALL: estimated cost of 2 for {{.*}} fadd double
define amdgpu_kernel void @fadd_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr, double %b) #0 {
  %vec = load double, double addrspace(1)* %vaddr
  %add = fadd double %vec, %b
  store double %add, double addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_v2f64'
; FASTF64: estimated cost of 4 for {{.*}} fadd <2 x double>
; SLOWF64: estimated cost of 8 for {{.*}} fadd <2 x double>
; SIZEALL: estimated cost of 4 for {{.*}} fadd <2 x double>
define amdgpu_kernel void @fadd_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %vaddr, <2 x double> %b) #0 {
  %vec = load <2 x double>, <2 x double> addrspace(1)* %vaddr
  %add = fadd <2 x double> %vec, %b
  store <2 x double> %add, <2 x double> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_v3f64'
; FASTF64: estimated cost of 6 for {{.*}} fadd <3 x double>
; SLOWF64: estimated cost of 12 for {{.*}} fadd <3 x double>
; SIZEALL: estimated cost of 6 for {{.*}} fadd <3 x double>
define amdgpu_kernel void @fadd_v3f64(<3 x double> addrspace(1)* %out, <3 x double> addrspace(1)* %vaddr, <3 x double> %b) #0 {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %vaddr
  %add = fadd <3 x double> %vec, %b
  store <3 x double> %add, <3 x double> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_f16'
; ALL: estimated cost of 1 for {{.*}} fadd half
define amdgpu_kernel void @fadd_f16(half addrspace(1)* %out, half addrspace(1)* %vaddr, half %b) #0 {
  %vec = load half, half addrspace(1)* %vaddr
  %add = fadd half %vec, %b
  store half %add, half addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_v2f16'
; SLOWF16: estimated cost of 2 for {{.*}} fadd <2 x half>
; FASTF16: estimated cost of 1 for {{.*}} fadd <2 x half>
define amdgpu_kernel void @fadd_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr, <2 x half> %b) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %add = fadd <2 x half> %vec, %b
  store <2 x half> %add, <2 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_v3f16'
; SLOWF16: estimated cost of 4 for {{.*}} fadd <3 x half>
; FASTF16: estimated cost of 2 for {{.*}} fadd <3 x half>
define amdgpu_kernel void @fadd_v3f16(<3 x half> addrspace(1)* %out, <3 x half> addrspace(1)* %vaddr, <3 x half> %b) #0 {
  %vec = load <3 x half>, <3 x half> addrspace(1)* %vaddr
  %add = fadd <3 x half> %vec, %b
  store <3 x half> %add, <3 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fadd_v4f16'
; SLOWF16: estimated cost of 4 for {{.*}} fadd <4 x half>
; FASTF16: estimated cost of 2 for {{.*}} fadd <4 x half>
define amdgpu_kernel void @fadd_v4f16(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %vaddr, <4 x half> %b) #0 {
  %vec = load <4 x half>, <4 x half> addrspace(1)* %vaddr
  %add = fadd <4 x half> %vec, %b
  store <4 x half> %add, <4 x half> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
