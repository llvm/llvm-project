; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=hawaii -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,CIFASTF64,NOFP32DENORM,NOFP16,NOFP16-NOFP32DENORM %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,CISLOWF64,NOFP32DENORM,NOFP16,NOFP16-NOFP32DENORM  %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=tahiti -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIFASTF64,NOFP32DENORM,NOFP16,NOFP16-NOFP32DENORM  %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=verde -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SISLOWF64,NOFP32DENORM,NOFP16,NOFP16-NOFP32DENORM  %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=hawaii -mattr=+fp32-denormals < %s | FileCheck -check-prefixes=ALL,FP32DENORMS,SLOWFP32DENORMS,NOFP16,NOFP16-FP32DENORM %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+fp32-denormals < %s | FileCheck -check-prefixes=ALL,FP32DENORMS,FASTFP32DENORMS,FP16 %s

; ALL: 'fdiv_f32'
; NOFP32DENORM: estimated cost of 12 for {{.*}} fdiv float
; FP32DENORMS: estimated cost of 10 for {{.*}} fdiv float
define amdgpu_kernel void @fdiv_f32(float addrspace(1)* %out, float addrspace(1)* %vaddr, float %b) #0 {
  %vec = load float, float addrspace(1)* %vaddr
  %add = fdiv float %vec, %b
  store float %add, float addrspace(1)* %out
  ret void
}

; ALL: 'fdiv_v2f32'
; NOFP32DENORM: estimated cost of 24 for {{.*}} fdiv <2 x float>
; FP32DENORMS: estimated cost of 20 for {{.*}} fdiv <2 x float>
define amdgpu_kernel void @fdiv_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr, <2 x float> %b) #0 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %add = fdiv <2 x float> %vec, %b
  store <2 x float> %add, <2 x float> addrspace(1)* %out
  ret void
}

; ALL: 'fdiv_v3f32'
; NOFP32DENORM: estimated cost of 36 for {{.*}} fdiv <3 x float>
; FP32DENORMS: estimated cost of 30 for {{.*}} fdiv <3 x float>
define amdgpu_kernel void @fdiv_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %vaddr, <3 x float> %b) #0 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %vaddr
  %add = fdiv <3 x float> %vec, %b
  store <3 x float> %add, <3 x float> addrspace(1)* %out
  ret void
}

; ALL: 'fdiv_f64'
; CIFASTF64: estimated cost of 29 for {{.*}} fdiv double
; CISLOWF64: estimated cost of 33 for {{.*}} fdiv double
; SIFASTF64: estimated cost of 32 for {{.*}} fdiv double
; SISLOWF64: estimated cost of 36 for {{.*}} fdiv double
define amdgpu_kernel void @fdiv_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr, double %b) #0 {
  %vec = load double, double addrspace(1)* %vaddr
  %add = fdiv double %vec, %b
  store double %add, double addrspace(1)* %out
  ret void
}

; ALL: 'fdiv_v2f64'
; CIFASTF64: estimated cost of 58 for {{.*}} fdiv <2 x double>
; CISLOWF64: estimated cost of 66 for {{.*}} fdiv <2 x double>
; SIFASTF64: estimated cost of 64 for {{.*}} fdiv <2 x double>
; SISLOWF64: estimated cost of 72 for {{.*}} fdiv <2 x double>
define amdgpu_kernel void @fdiv_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %vaddr, <2 x double> %b) #0 {
  %vec = load <2 x double>, <2 x double> addrspace(1)* %vaddr
  %add = fdiv <2 x double> %vec, %b
  store <2 x double> %add, <2 x double> addrspace(1)* %out
  ret void
}

; ALL: 'fdiv_v3f64'
; CIFASTF64: estimated cost of 87 for {{.*}} fdiv <3 x double>
; CISLOWF64: estimated cost of 99 for {{.*}} fdiv <3 x double>
; SIFASTF64: estimated cost of 96 for {{.*}} fdiv <3 x double>
; SISLOWF64: estimated cost of 108 for {{.*}} fdiv <3 x double>
define amdgpu_kernel void @fdiv_v3f64(<3 x double> addrspace(1)* %out, <3 x double> addrspace(1)* %vaddr, <3 x double> %b) #0 {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %vaddr
  %add = fdiv <3 x double> %vec, %b
  store <3 x double> %add, <3 x double> addrspace(1)* %out
  ret void
}

; ALL: 'fdiv_f16'
; NOFP16-NOFP32DENORM: estimated cost of 12 for {{.*}} fdiv half
; NOFP16-FP32DENORM: estimated cost of 10 for {{.*}} fdiv half
; FP16: estimated cost of 10 for {{.*}} fdiv half
define amdgpu_kernel void @fdiv_f16(half addrspace(1)* %out, half addrspace(1)* %vaddr, half %b) #0 {
  %vec = load half, half addrspace(1)* %vaddr
  %add = fdiv half %vec, %b
  store half %add, half addrspace(1)* %out
  ret void
}

; ALL: 'fdiv_v2f16'
; NOFP16-NOFP32DENORM: estimated cost of 24 for {{.*}} fdiv <2 x half>
; NOFP16-FP32DENORM: estimated cost of 20 for {{.*}} fdiv <2 x half>
; FP16: estimated cost of 20 for {{.*}} fdiv <2 x half>
define amdgpu_kernel void @fdiv_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr, <2 x half> %b) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %add = fdiv <2 x half> %vec, %b
  store <2 x half> %add, <2 x half> addrspace(1)* %out
  ret void
}

; ALL: 'fdiv_v4f16'
; NOFP16-NOFP32DENORM: estimated cost of 48 for {{.*}} fdiv <4 x half>
; NOFP16-FP32DENORM: estimated cost of 40 for {{.*}} fdiv <4 x half>
; FP16: estimated cost of 40 for {{.*}} fdiv <4 x half>
define amdgpu_kernel void @fdiv_v4f16(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %vaddr, <4 x half> %b) #0 {
  %vec = load <4 x half>, <4 x half> addrspace(1)* %vaddr
  %add = fdiv <4 x half> %vec, %b
  store <4 x half> %add, <4 x half> addrspace(1)* %out
  ret void
}

; ALL: 'rcp_f32'
; NOFP32DENORM: estimated cost of 3 for {{.*}} fdiv float
; SLOWFP32DENORMS: estimated cost of 10 for {{.*}} fdiv float
; FASTFP32DENORMS: estimated cost of 10 for {{.*}} fdiv float
define amdgpu_kernel void @rcp_f32(float addrspace(1)* %out, float addrspace(1)* %vaddr) #0 {
  %vec = load float, float addrspace(1)* %vaddr
  %add = fdiv float 1.0, %vec
  store float %add, float addrspace(1)* %out
  ret void
}

; ALL: 'rcp_f16'
; NOFP16-NOFP32DENORM: estimated cost of 3 for {{.*}} fdiv half
; NOFP16-FP32DENORM: estimated cost of 10 for {{.*}} fdiv half
; FP16: estimated cost of 3 for {{.*}} fdiv half
define amdgpu_kernel void @rcp_f16(half addrspace(1)* %out, half addrspace(1)* %vaddr) #0 {
  %vec = load half, half addrspace(1)* %vaddr
  %add = fdiv half 1.0, %vec
  store half %add, half addrspace(1)* %out
  ret void
}

; ALL: 'rcp_f64'
; CIFASTF64: estimated cost of 29 for {{.*}} fdiv double
; CISLOWF64: estimated cost of 33 for {{.*}} fdiv double
; SIFASTF64: estimated cost of 32 for {{.*}} fdiv double
; SISLOWF64: estimated cost of 36 for {{.*}} fdiv double
define amdgpu_kernel void @rcp_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr) #0 {
  %vec = load double, double addrspace(1)* %vaddr
  %add = fdiv double 1.0, %vec
  store double %add, double addrspace(1)* %out
  ret void
}

; ALL: 'rcp_v2f32'
; NOFP32DENORM: estimated cost of 6 for {{.*}} fdiv <2 x float>
; SLOWFP32DENORMS: estimated cost of 20 for {{.*}} fdiv <2 x float>
; FASTFP32DENORMS: estimated cost of 20 for {{.*}} fdiv <2 x float>
define amdgpu_kernel void @rcp_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %add = fdiv <2 x float> <float 1.0, float 1.0>, %vec
  store <2 x float> %add, <2 x float> addrspace(1)* %out
  ret void
}

; ALL: 'rcp_v2f16'
; NOFP16-NOFP32DENORM: estimated cost of 6 for {{.*}} fdiv <2 x half>
; NOFP16-FP32DENORM: estimated cost of 20 for {{.*}} fdiv <2 x half>
; FP16: estimated cost of 6 for {{.*}} fdiv <2 x half>
define amdgpu_kernel void @rcp_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %add = fdiv <2 x half> <half 1.0, half 1.0>, %vec
  store <2 x half> %add, <2 x half> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
