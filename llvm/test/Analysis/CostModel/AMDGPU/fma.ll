; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900  -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FASTF64,FASTF32,FASTF16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOWF64,SLOWF32,SLOWF16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZEF16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZENOF16 %s

; ALL-LABEL: 'fma_f32'
; SLOWF32: estimated cost of 4 for {{.*}} call float @llvm.fma.f32
; FASTF32: estimated cost of 2 for {{.*}} call float @llvm.fma.f32
; SIZEALL: estimated cost of 2 for {{.*}} call float @llvm.fma.f32
define amdgpu_kernel void @fma_f32(float addrspace(1)* %out, float addrspace(1)* %vaddr) #0 {
  %vec = load float, float addrspace(1)* %vaddr
  %fma = call float @llvm.fma.f32(float %vec, float %vec, float %vec) #1
  store float %fma, float addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_v2f32'
; SLOWF32: estimated cost of 8 for {{.*}} call <2 x float> @llvm.fma.v2f32
; FASTF32: estimated cost of 4 for {{.*}} call <2 x float> @llvm.fma.v2f32
; SIZEALL: estimated cost of 4 for {{.*}} call <2 x float> @llvm.fma.v2f32
define amdgpu_kernel void @fma_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %fma = call <2 x float> @llvm.fma.v2f32(<2 x float> %vec, <2 x float> %vec, <2 x float> %vec) #1
  store <2 x float> %fma, <2 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_v3f32'
; SLOWF32: estimated cost of 12 for {{.*}} call <3 x float> @llvm.fma.v3f32
; FASTF32: estimated cost of 6 for {{.*}} call <3 x float> @llvm.fma.v3f32
; SIZEALL: estimated cost of 6 for {{.*}} call <3 x float> @llvm.fma.v3f32
define amdgpu_kernel void @fma_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %vaddr) #0 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %vaddr
  %fma = call <3 x float> @llvm.fma.v3f32(<3 x float> %vec, <3 x float> %vec, <3 x float> %vec) #1
  store <3 x float> %fma, <3 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_v5f32'
; SLOWF32: estimated cost of 20 for {{.*}} call <5 x float> @llvm.fma.v5f32
; FASTF32: estimated cost of 10 for {{.*}} call <5 x float> @llvm.fma.v5f32
; SIZEALL: estimated cost of 10 for {{.*}} call <5 x float> @llvm.fma.v5f32
define amdgpu_kernel void @fma_v5f32(<5 x float> addrspace(1)* %out, <5 x float> addrspace(1)* %vaddr) #0 {
  %vec = load <5 x float>, <5 x float> addrspace(1)* %vaddr
  %fma = call <5 x float> @llvm.fma.v5f32(<5 x float> %vec, <5 x float> %vec, <5 x float> %vec) #1
  store <5 x float> %fma, <5 x float> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_f64'
; SLOWF64: estimated cost of 4 for {{.*}} call double @llvm.fma.f64
; FASTF64: estimated cost of 2 for {{.*}} call double @llvm.fma.f64
; SIZEALL: estimated cost of 2 for {{.*}} call double @llvm.fma.f64
define amdgpu_kernel void @fma_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr) #0 {
  %vec = load double, double addrspace(1)* %vaddr
  %fma = call double @llvm.fma.f64(double %vec, double %vec, double %vec) #1
  store double %fma, double addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_v2f64'
; SLOWF64: estimated cost of 8 for {{.*}} call <2 x double> @llvm.fma.v2f64
; FASTF64: estimated cost of 4 for {{.*}} call <2 x double> @llvm.fma.v2f64
; SIZEALL: estimated cost of 4 for {{.*}} call <2 x double> @llvm.fma.v2f64
define amdgpu_kernel void @fma_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x double>, <2 x double> addrspace(1)* %vaddr
  %fma = call <2 x double> @llvm.fma.v2f64(<2 x double> %vec, <2 x double> %vec, <2 x double> %vec) #1
  store <2 x double> %fma, <2 x double> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_v3f64'
; SLOWF64: estimated cost of 12 for {{.*}} call <3 x double> @llvm.fma.v3f64
; FASTF64: estimated cost of 6 for {{.*}} call <3 x double> @llvm.fma.v3f64
; SIZEALL: estimated cost of 6 for {{.*}} call <3 x double> @llvm.fma.v3f64
define amdgpu_kernel void @fma_v3f64(<3 x double> addrspace(1)* %out, <3 x double> addrspace(1)* %vaddr) #0 {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %vaddr
  %fma = call <3 x double> @llvm.fma.v3f64(<3 x double> %vec, <3 x double> %vec, <3 x double> %vec) #1
  store <3 x double> %fma, <3 x double> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_f16'
; SLOWF16: estimated cost of 4 for {{.*}} call half @llvm.fma.f16
; FASTF16: estimated cost of 2 for {{.*}} call half @llvm.fma.f16
; SIZEALL: estimated cost of 2 for {{.*}} call half @llvm.fma.f16
define amdgpu_kernel void @fma_f16(half addrspace(1)* %out, half addrspace(1)* %vaddr) #0 {
  %vec = load half, half addrspace(1)* %vaddr
  %fma = call half @llvm.fma.f16(half %vec, half %vec, half %vec) #1
  store half %fma, half addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_v2f16'
; SLOWF16: estimated cost of 8 for {{.*}} call <2 x half> @llvm.fma.v2f16
; FASTF16: estimated cost of 2 for {{.*}} call <2 x half> @llvm.fma.v2f16
; SIZEF16: estimated cost of 2 for {{.*}} call <2 x half> @llvm.fma.v2f16
; SIZENOF16: estimated cost of 4 for {{.*}} call <2 x half> @llvm.fma.v2f16
define amdgpu_kernel void @fma_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %fma = call <2 x half> @llvm.fma.v2f16(<2 x half> %vec, <2 x half> %vec, <2 x half> %vec) #1
  store <2 x half> %fma, <2 x half> addrspace(1)* %out
  ret void
}

; ALL-LABEL: 'fma_v3f16'
; SLOWF16: estimated cost of 16 for {{.*}} call <3 x half> @llvm.fma.v3f16
; FASTF16: estimated cost of 4 for {{.*}} call <3 x half> @llvm.fma.v3f16
; SIZEF16: estimated cost of 4 for {{.*}} call <3 x half> @llvm.fma.v3f16
; SIZENOF16: estimated cost of 8 for {{.*}} call <3 x half> @llvm.fma.v3f16
define amdgpu_kernel void @fma_v3f16(<3 x half> addrspace(1)* %out, <3 x half> addrspace(1)* %vaddr) #0 {
  %vec = load <3 x half>, <3 x half> addrspace(1)* %vaddr
  %fma = call <3 x half> @llvm.fma.v3f16(<3 x half> %vec, <3 x half> %vec, <3 x half> %vec) #1
  store <3 x half> %fma, <3 x half> addrspace(1)* %out
  ret void
}

declare float @llvm.fma.f32(float, float, float) #1
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>) #1
declare <3 x float> @llvm.fma.v3f32(<3 x float>, <3 x float>, <3 x float>) #1
declare <5 x float> @llvm.fma.v5f32(<5 x float>, <5 x float>, <5 x float>) #1

declare double @llvm.fma.f64(double, double, double) #1
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>) #1
declare <3 x double> @llvm.fma.v3f64(<3 x double>, <3 x double>, <3 x double>) #1

declare half @llvm.fma.f16(half, half, half) #1
declare <2 x half> @llvm.fma.v2f16(<2 x half>, <2 x half>, <2 x half>) #1
declare <3 x half> @llvm.fma.v3f16(<3 x half>, <3 x half>, <3 x half>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
