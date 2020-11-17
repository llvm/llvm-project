; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=hawaii < %s | FileCheck -check-prefixes=ALL,THRPTALL,CIFASTF64,NOFP16 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri < %s | FileCheck -check-prefixes=ALL,THRPTALL,CISLOWF64,NOFP16  %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti < %s | FileCheck -check-prefixes=ALL,THRPTALL,SIFASTF64,NOFP16  %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-mesa-mesa3d -mcpu=verde < %s | FileCheck -check-prefixes=ALL,THRPTALL,SISLOWF64,NOFP16  %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=ALL,THRPTALL,FP16,CISLOWF64 %s

; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=hawaii < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZECI,SIZENOF16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZECI,SIZENOF16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZESI,SIZENOF16  %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-mesa-mesa3d -mcpu=verde < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZESI,SIZENOF16  %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZECI,SIZEF16 %s

; ALL-LABEL: 'fdiv_f32_ieee'
; THRPTALL: estimated cost of 14 for {{.*}} fdiv float
; SIZEALL: estimated cost of 12 for {{.*}} fdiv float
define amdgpu_kernel void @fdiv_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %vaddr, float %b) #0 {
  %vec = load float, float addrspace(1)* %vaddr
  %add = fdiv float %vec, %b
  store float %add, float addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_f32_ftzdaz'
; THRPTALL: estimated cost of 16 for {{.*}} fdiv float
; SIZEALL: estimated cost of 14 for {{.*}} fdiv float
define amdgpu_kernel void @fdiv_f32_ftzdaz(float addrspace(1)* %out, float addrspace(1)* %vaddr, float %b) #1 {
  %vec = load float, float addrspace(1)* %vaddr
  %add = fdiv float %vec, %b
  store float %add, float addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v2f32_ieee'
; THRPTALL: estimated cost of 28 for {{.*}} fdiv <2 x float>
; SIZEALL: estimated cost of 24 for {{.*}} fdiv <2 x float>
define amdgpu_kernel void @fdiv_v2f32_ieee(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr, <2 x float> %b) #0 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %add = fdiv <2 x float> %vec, %b
  store <2 x float> %add, <2 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v2f32_ftzdaz'
; THRPTALL: estimated cost of 32 for {{.*}} fdiv <2 x float>
; SIZEALL: estimated cost of 28 for {{.*}} fdiv <2 x float>
define amdgpu_kernel void @fdiv_v2f32_ftzdaz(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr, <2 x float> %b) #1 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %add = fdiv <2 x float> %vec, %b
  store <2 x float> %add, <2 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v3f32_ieee'
; THRPTALL: estimated cost of 42 for {{.*}} fdiv <3 x float>
; SIZEALL: estimated cost of 36 for {{.*}} fdiv <3 x float>
define amdgpu_kernel void @fdiv_v3f32_ieee(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %vaddr, <3 x float> %b) #0 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %vaddr
  %add = fdiv <3 x float> %vec, %b
  store <3 x float> %add, <3 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v3f32_ftzdaz'
; THRPTALL: estimated cost of 48 for {{.*}} fdiv <3 x float>
; SIZEALL: estimated cost of 42 for {{.*}} fdiv <3 x float>
define amdgpu_kernel void @fdiv_v3f32_ftzdaz(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %vaddr, <3 x float> %b) #1 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %vaddr
  %add = fdiv <3 x float> %vec, %b
  store <3 x float> %add, <3 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v5f32_ieee'
; THRPTALL: estimated cost of 70 for {{.*}} fdiv <5 x float>
; SIZEALL: estimated cost of 60 for {{.*}} fdiv <5 x float>
define amdgpu_kernel void @fdiv_v5f32_ieee(<5 x float> addrspace(1)* %out, <5 x float> addrspace(1)* %vaddr, <5 x float> %b) #0 {
  %vec = load <5 x float>, <5 x float> addrspace(1)* %vaddr
  %add = fdiv <5 x float> %vec, %b
  store <5 x float> %add, <5 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v5f32_ftzdaz'
; THRPTALL: estimated cost of 80 for {{.*}} fdiv <5 x float>
; SIZEALL: estimated cost of 70 for {{.*}} fdiv <5 x float>
define amdgpu_kernel void @fdiv_v5f32_ftzdaz(<5 x float> addrspace(1)* %out, <5 x float> addrspace(1)* %vaddr, <5 x float> %b) #1 {
  %vec = load <5 x float>, <5 x float> addrspace(1)* %vaddr
  %add = fdiv <5 x float> %vec, %b
  store <5 x float> %add, <5 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_f64'
; CIFASTF64: estimated cost of 24 for {{.*}} fdiv double
; CISLOWF64: estimated cost of 38 for {{.*}} fdiv double
; SIFASTF64: estimated cost of 27 for {{.*}} fdiv double
; SISLOWF64: estimated cost of 41 for {{.*}} fdiv double
; SIZECI: estimated cost of 22 for {{.*}} fdiv double
; SIZESI: estimated cost of 25 for {{.*}} fdiv double
define amdgpu_kernel void @fdiv_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr, double %b) #0 {
  %vec = load double, double addrspace(1)* %vaddr
  %add = fdiv double %vec, %b
  store double %add, double addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v2f64'
; CIFASTF64: estimated cost of 48 for {{.*}} fdiv <2 x double>
; CISLOWF64: estimated cost of 76 for {{.*}} fdiv <2 x double>
; SIFASTF64: estimated cost of 54 for {{.*}} fdiv <2 x double>
; SISLOWF64: estimated cost of 82 for {{.*}} fdiv <2 x double>
; SIZECI: estimated cost of 44 for {{.*}} fdiv <2 x double>
; SIZESI: estimated cost of 50 for {{.*}} fdiv <2 x double>
define amdgpu_kernel void @fdiv_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %vaddr, <2 x double> %b) #0 {
  %vec = load <2 x double>, <2 x double> addrspace(1)* %vaddr
  %add = fdiv <2 x double> %vec, %b
  store <2 x double> %add, <2 x double> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v3f64'
; CIFASTF64: estimated cost of 72 for {{.*}} fdiv <3 x double>
; CISLOWF64: estimated cost of 114 for {{.*}} fdiv <3 x double>
; SIFASTF64: estimated cost of 81 for {{.*}} fdiv <3 x double>
; SISLOWF64: estimated cost of 123 for {{.*}} fdiv <3 x double>
; SIZECI: estimated cost of 66 for {{.*}} fdiv <3 x double>
; SIZESI: estimated cost of 75 for {{.*}} fdiv <3 x double>
define amdgpu_kernel void @fdiv_v3f64(<3 x double> addrspace(1)* %out, <3 x double> addrspace(1)* %vaddr, <3 x double> %b) #0 {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %vaddr
  %add = fdiv <3 x double> %vec, %b
  store <3 x double> %add, <3 x double> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_f16_f32_ieee'
; NOFP16: estimated cost of 14 for {{.*}} fdiv half
; FP16: estimated cost of 12 for {{.*}} fdiv half
; SIZENOF16: estimated cost of 12 for {{.*}} fdiv half
; SIZEF16: estimated cost of 8 for {{.*}} fdiv half
define amdgpu_kernel void @fdiv_f16_f32_ieee(half addrspace(1)* %out, half addrspace(1)* %vaddr, half %b) #0 {
  %vec = load half, half addrspace(1)* %vaddr
  %add = fdiv half %vec, %b
  store half %add, half addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_f16_f32_ftzdaz'
; NOFP16: estimated cost of 16 for {{.*}} fdiv half
; FP16: estimated cost of 12 for {{.*}} fdiv half
; SIZENOF16: estimated cost of 14 for {{.*}} fdiv half
; SIZEF16: estimated cost of 8 for {{.*}} fdiv half
define amdgpu_kernel void @fdiv_f16_f32_ftzdaz(half addrspace(1)* %out, half addrspace(1)* %vaddr, half %b) #1 {
  %vec = load half, half addrspace(1)* %vaddr
  %add = fdiv half %vec, %b
  store half %add, half addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v2f16_f32_ieee'
; NOFP16: estimated cost of 28 for {{.*}} fdiv <2 x half>
; FP16: estimated cost of 24 for {{.*}} fdiv <2 x half>
; SIZENOF16: estimated cost of 24 for {{.*}} fdiv <2 x half>
; SIZEF16: estimated cost of 16 for {{.*}} fdiv <2 x half>
define amdgpu_kernel void @fdiv_v2f16_f32_ieee(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr, <2 x half> %b) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %add = fdiv <2 x half> %vec, %b
  store <2 x half> %add, <2 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v2f16_f32_ftzdaz'
; NOFP16: estimated cost of 32 for {{.*}} fdiv <2 x half>
; FP16: estimated cost of 24 for {{.*}} fdiv <2 x half>
; SIZENOF16: estimated cost of 28 for {{.*}} fdiv <2 x half>
; SIZEF16: estimated cost of 16 for {{.*}} fdiv <2 x half>
define amdgpu_kernel void @fdiv_v2f16_f32_ftzdaz(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr, <2 x half> %b) #1 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %add = fdiv <2 x half> %vec, %b
  store <2 x half> %add, <2 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v4f16_f32_ieee'
; NOFP16: estimated cost of 56 for {{.*}} fdiv <4 x half>
; FP16: estimated cost of 48 for {{.*}} fdiv <4 x half>
; SIZENOF16: estimated cost of 48 for {{.*}} fdiv <4 x half>
; SIZEF16: estimated cost of 32 for {{.*}} fdiv <4 x half>
define amdgpu_kernel void @fdiv_v4f16_f32_ieee(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %vaddr, <4 x half> %b) #0 {
  %vec = load <4 x half>, <4 x half> addrspace(1)* %vaddr
  %add = fdiv <4 x half> %vec, %b
  store <4 x half> %add, <4 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fdiv_v4f16_f32_ftzdaz'
; NOFP16: estimated cost of 64 for {{.*}} fdiv <4 x half>
; FP16: estimated cost of 48 for {{.*}} fdiv <4 x half>
; SIZENOF16: estimated cost of 56 for {{.*}} fdiv <4 x half>
; SIZEF16: estimated cost of 32 for {{.*}} fdiv <4 x half>
define amdgpu_kernel void @fdiv_v4f16_f32_ftzdaz(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %vaddr, <4 x half> %b) #1 {
  %vec = load <4 x half>, <4 x half> addrspace(1)* %vaddr
  %add = fdiv <4 x half> %vec, %b
  store <4 x half> %add, <4 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_f32_ieee'
; THRPTALL: estimated cost of 14 for {{.*}} fdiv float
; SIZEALL: estimated cost of 12 for {{.*}} fdiv float
define amdgpu_kernel void @rcp_f32_ieee(float addrspace(1)* %out, float addrspace(1)* %vaddr) #0 {
  %vec = load float, float addrspace(1)* %vaddr
  %add = fdiv float 1.0, %vec
  store float %add, float addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_f32_ftzdaz'
; THRPTALL: estimated cost of 4 for {{.*}} fdiv float
; SIZEALL: estimated cost of 2 for {{.*}} fdiv float
define amdgpu_kernel void @rcp_f32_ftzdaz(float addrspace(1)* %out, float addrspace(1)* %vaddr) #1 {
  %vec = load float, float addrspace(1)* %vaddr
  %add = fdiv float 1.0, %vec
  store float %add, float addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_f16_f32_ieee'
; NOFP16: estimated cost of 14 for {{.*}} fdiv half
; FP16: estimated cost of 4 for {{.*}} fdiv half
; SIZENOF16: estimated cost of 12 for {{.*}} fdiv half
; SIZEF16: estimated cost of 2 for {{.*}} fdiv half
define amdgpu_kernel void @rcp_f16_f32_ieee(half addrspace(1)* %out, half addrspace(1)* %vaddr) #0 {
  %vec = load half, half addrspace(1)* %vaddr
  %add = fdiv half 1.0, %vec
  store half %add, half addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_f16_f32_ftzdaz'
; THRPTALL: estimated cost of 4 for {{.*}} fdiv half
; SIZEALL: estimated cost of 2 for {{.*}} fdiv half
define amdgpu_kernel void @rcp_f16_f32_ftzdaz(half addrspace(1)* %out, half addrspace(1)* %vaddr) #1 {
  %vec = load half, half addrspace(1)* %vaddr
  %add = fdiv half 1.0, %vec
  store half %add, half addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_f64'
; CIFASTF64: estimated cost of 24 for {{.*}} fdiv double
; CISLOWF64: estimated cost of 38 for {{.*}} fdiv double
; SIFASTF64: estimated cost of 27 for {{.*}} fdiv double
; SISLOWF64: estimated cost of 41 for {{.*}} fdiv double
; SIZECI: estimated cost of 22 for {{.*}} fdiv double
; SIZESI: estimated cost of 25 for {{.*}} fdiv double
define amdgpu_kernel void @rcp_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr) #0 {
  %vec = load double, double addrspace(1)* %vaddr
  %add = fdiv double 1.0, %vec
  store double %add, double addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_v2f32_ieee'
; THRPTALL: estimated cost of 28 for {{.*}} fdiv <2 x float>
; SIZEALL: estimated cost of 24 for {{.*}} fdiv <2 x float>
define amdgpu_kernel void @rcp_v2f32_ieee(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %add = fdiv <2 x float> <float 1.0, float 1.0>, %vec
  store <2 x float> %add, <2 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_v2f32_ftzdaz'
; THRPTALL: estimated cost of 8 for {{.*}} fdiv <2 x float>
; SIZEALL: estimated cost of 4 for {{.*}} fdiv <2 x float>
define amdgpu_kernel void @rcp_v2f32_ftzdaz(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr) #1 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %add = fdiv <2 x float> <float 1.0, float 1.0>, %vec
  store <2 x float> %add, <2 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_v2f16_f32_ieee'
; NOFP16: estimated cost of 28 for {{.*}} fdiv <2 x half>
; FP16: estimated cost of 8 for {{.*}} fdiv <2 x half>
; SIZENOF16: estimated cost of 24 for {{.*}} fdiv <2 x half>
; SIZEF16: estimated cost of 4 for {{.*}} fdiv <2 x half>
define amdgpu_kernel void @rcp_v2f16_f32_ieee(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %add = fdiv <2 x half> <half 1.0, half 1.0>, %vec
  store <2 x half> %add, <2 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'rcp_v2f16_f32_ftzdaz'
; THRPTALL: estimated cost of 8 for {{.*}} fdiv <2 x half>
; SIZEALL: estimated cost of 4 for {{.*}} fdiv <2 x half>
define amdgpu_kernel void @rcp_v2f16_f32_ftzdaz(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr) #1 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %add = fdiv <2 x half> <half 1.0, half 1.0>, %vec
  store <2 x half> %add, <2 x half> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind "denormal-fp-math-f32"="ieee,ieee" }
attributes #1 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
