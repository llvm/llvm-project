; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s
; END.

; CHECK-LABEL: 'fabs_f32'
; CHECK: estimated cost of 0 for {{.*}} call float @llvm.fabs.f32
; CHECK: estimated cost of 0 for {{.*}} call <2 x float> @llvm.fabs.v2f32
; CHECK: estimated cost of 0 for {{.*}} call <3 x float> @llvm.fabs.v3f32
; CHECK: estimated cost of 0 for {{.*}} call <5 x float> @llvm.fabs.v5f32
define amdgpu_kernel void @fabs_f32() #0 {
  %f32 = call float @llvm.fabs.f32(float undef) #1
  %v2f32 = call <2 x float> @llvm.fabs.v2f32(<2 x float> undef) #1
  %v3f32 = call <3 x float> @llvm.fabs.v3f32(<3 x float> undef) #1
  %v5f32 = call <5 x float> @llvm.fabs.v5f32(<5 x float> undef) #1
  ret void
}

; CHECK-LABEL: 'fabs_f64'
; CHECK: estimated cost of 0 for {{.*}} call double @llvm.fabs.f64
; CHECK: estimated cost of 0 for {{.*}} call <2 x double> @llvm.fabs.v2f64
; CHECK: estimated cost of 0 for {{.*}} call <3 x double> @llvm.fabs.v3f64
define amdgpu_kernel void @fabs_f64() #0 {
  %f64 = call double @llvm.fabs.f64(double undef) #1
  %v2f64 = call <2 x double> @llvm.fabs.v2f64(<2 x double> undef) #1
  %v3f64 = call <3 x double> @llvm.fabs.v3f64(<3 x double> undef) #1
  ret void
}

; CHECK-LABEL: 'fabs_f16'
; CHECK: estimated cost of 0 for {{.*}} call half @llvm.fabs.f16
; CHECK: estimated cost of 0 for {{.*}} call <2 x half> @llvm.fabs.v2f16
; CHECK: estimated cost of 0 for {{.*}} call <3 x half> @llvm.fabs.v3f16
define amdgpu_kernel void @fabs_f16() #0 {
  %f16 = call half @llvm.fabs.f16(half undef) #1
  %v2f16 = call <2 x half> @llvm.fabs.v2f16(<2 x half> undef) #1
  %v3f16 = call <3 x half> @llvm.fabs.v3f16(<3 x half> undef) #1
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
