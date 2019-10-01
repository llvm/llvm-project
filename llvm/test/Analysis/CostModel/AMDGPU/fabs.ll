; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; CHECK: 'fabs_f32'
; CHECK: estimated cost of 0 for {{.*}} call float @llvm.fabs.f32
define amdgpu_kernel void @fabs_f32(float addrspace(1)* %out, float addrspace(1)* %vaddr) #0 {
  %vec = load float, float addrspace(1)* %vaddr
  %fabs = call float @llvm.fabs.f32(float %vec) #1
  store float %fabs, float addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_v2f32'
; CHECK: estimated cost of 0 for {{.*}} call <2 x float> @llvm.fabs.v2f32
define amdgpu_kernel void @fabs_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x float>, <2 x float> addrspace(1)* %vaddr
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %vec) #1
  store <2 x float> %fabs, <2 x float> addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_v3f32'
; CHECK: estimated cost of 0 for {{.*}} call <3 x float> @llvm.fabs.v3f32
define amdgpu_kernel void @fabs_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %vaddr) #0 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %vaddr
  %fabs = call <3 x float> @llvm.fabs.v3f32(<3 x float> %vec) #1
  store <3 x float> %fabs, <3 x float> addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_v5f32'
; CHECK: estimated cost of 0 for {{.*}} call <5 x float> @llvm.fabs.v5f32
define amdgpu_kernel void @fabs_v5f32(<5 x float> addrspace(1)* %out, <5 x float> addrspace(1)* %vaddr) #0 {
  %vec = load <5 x float>, <5 x float> addrspace(1)* %vaddr
  %fabs = call <5 x float> @llvm.fabs.v5f32(<5 x float> %vec) #1
  store <5 x float> %fabs, <5 x float> addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_f64'
; CHECK: estimated cost of 0 for {{.*}} call double @llvm.fabs.f64
define amdgpu_kernel void @fabs_f64(double addrspace(1)* %out, double addrspace(1)* %vaddr) #0 {
  %vec = load double, double addrspace(1)* %vaddr
  %fabs = call double @llvm.fabs.f64(double %vec) #1
  store double %fabs, double addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_v2f64'
; CHECK: estimated cost of 0 for {{.*}} call <2 x double> @llvm.fabs.v2f64
define amdgpu_kernel void @fabs_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x double>, <2 x double> addrspace(1)* %vaddr
  %fabs = call <2 x double> @llvm.fabs.v2f64(<2 x double> %vec) #1
  store <2 x double> %fabs, <2 x double> addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_v3f64'
; CHECK: estimated cost of 0 for {{.*}} call <3 x double> @llvm.fabs.v3f64
define amdgpu_kernel void @fabs_v3f64(<3 x double> addrspace(1)* %out, <3 x double> addrspace(1)* %vaddr) #0 {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %vaddr
  %fabs = call <3 x double> @llvm.fabs.v3f64(<3 x double> %vec) #1
  store <3 x double> %fabs, <3 x double> addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_f16'
; CHECK: estimated cost of 0 for {{.*}} call half @llvm.fabs.f16
define amdgpu_kernel void @fabs_f16(half addrspace(1)* %out, half addrspace(1)* %vaddr) #0 {
  %vec = load half, half addrspace(1)* %vaddr
  %fabs = call half @llvm.fabs.f16(half %vec) #1
  store half %fabs, half addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_v2f16'
; CHECK: estimated cost of 0 for {{.*}} call <2 x half> @llvm.fabs.v2f16
define amdgpu_kernel void @fabs_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %vaddr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(1)* %vaddr
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %vec) #1
  store <2 x half> %fabs, <2 x half> addrspace(1)* %out
  ret void
}

; CHECK: 'fabs_v3f16'
; CHECK: estimated cost of 0 for {{.*}} call <3 x half> @llvm.fabs.v3f16
define amdgpu_kernel void @fabs_v3f16(<3 x half> addrspace(1)* %out, <3 x half> addrspace(1)* %vaddr) #0 {
  %vec = load <3 x half>, <3 x half> addrspace(1)* %vaddr
  %fabs = call <3 x half> @llvm.fabs.v3f16(<3 x half> %vec) #1
  store <3 x half> %fabs, <3 x half> addrspace(1)* %out
  ret void
}

declare float @llvm.fabs.f32(float) #1
declare <2 x float> @llvm.fabs.v2f32(<2 x float>) #1
declare <3 x float> @llvm.fabs.v3f32(<3 x float>) #1
declare <5 x float> @llvm.fabs.v5f32(<5 x float>) #1

declare double @llvm.fabs.f64(double) #1
declare <2 x double> @llvm.fabs.v2f64(<2 x double>) #1
declare <3 x double> @llvm.fabs.v3f64(<3 x double>) #1

declare half @llvm.fabs.f16(half) #1
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #1
declare <3 x half> @llvm.fabs.v3f16(<3 x half>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
