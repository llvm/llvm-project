; RUN: llc -march=amdgcn -verify-machineinstrs < %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s

; This should have the exactly the same output as the test for rint,
; so no need to check anything.

declare float @llvm.nearbyint.f32(float) #0
declare <2 x float> @llvm.nearbyint.v2f32(<2 x float>) #0
declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>) #0
declare double @llvm.nearbyint.f64(double) #0
declare <2 x double> @llvm.nearbyint.v2f64(<2 x double>) #0
declare <4 x double> @llvm.nearbyint.v4f64(<4 x double>) #0


define amdgpu_kernel void @fnearbyint_f32(float addrspace(1)* %out, float %in) #1 {
entry:
  %0 = call float @llvm.nearbyint.f32(float %in)
  store float %0, float addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @fnearbyint_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) #1 {
entry:
  %0 = call <2 x float> @llvm.nearbyint.v2f32(<2 x float> %in)
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @fnearbyint_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) #1 {
entry:
  %0 = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %in)
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @nearbyint_f64(double addrspace(1)* %out, double %in) {
entry:
  %0 = call double @llvm.nearbyint.f64(double %in)
  store double %0, double addrspace(1)* %out
  ret void
}
define amdgpu_kernel void @nearbyint_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %in) {
entry:
  %0 = call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %in)
  store <2 x double> %0, <2 x double> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @nearbyint_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %in) {
entry:
  %0 = call <4 x double> @llvm.nearbyint.v4f64(<4 x double> %in)
  store <4 x double> %0, <4 x double> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind }
