; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900  -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FASTF64,FASTF32,FASTF16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOWF64,SLOWF32,SLOWF16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZEF16 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=ALL,SIZEALL,SIZENOF16 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx90a -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=GFX90A-FASTF64,FASTF16,PACKEDF32,ALL %s
; END.

; ALL-LABEL: 'fma_f32'
; SLOWF32: estimated cost of 4 for {{.*}} call float @llvm.fma.f32
; FASTF32: estimated cost of 2 for {{.*}} call float @llvm.fma.f32
; SIZEALL: estimated cost of 2 for {{.*}} call float @llvm.fma.f32
; SLOWF32: estimated cost of 8 for {{.*}} call <2 x float> @llvm.fma.v2f32
; PACKEDF32: estimated cost of 2 for {{.*}} call <2 x float> @llvm.fma.v2f32
; SIZEALL: estimated cost of 4 for {{.*}} call <2 x float> @llvm.fma.v2f32
; SLOWF32: estimated cost of 12 for {{.*}} call <3 x float> @llvm.fma.v3f32
; PACKEDF32: estimated cost of 4 for {{.*}} call <3 x float> @llvm.fma.v3f32
; SIZEALL: estimated cost of 6 for {{.*}} call <3 x float> @llvm.fma.v3f32
; SLOWF32: estimated cost of 20 for {{.*}} call <5 x float> @llvm.fma.v5f32
; PACKEDF32: estimated cost of 6 for {{.*}} call <5 x float> @llvm.fma.v5f32
; SIZEALL: estimated cost of 10 for {{.*}} call <5 x float> @llvm.fma.v5f32
define amdgpu_kernel void @fma_f32() #0 {
  %f32 = call float @llvm.fma.f32(float undef, float undef, float undef) #1
  %v2f32 = call <2 x float> @llvm.fma.v2f32(<2 x float> undef, <2 x float> undef, <2 x float> undef) #1
  %v3f32 = call <3 x float> @llvm.fma.v3f32(<3 x float> undef, <3 x float> undef, <3 x float> undef) #1
  %v5f32 = call <5 x float> @llvm.fma.v5f32(<5 x float> undef, <5 x float> undef, <5 x float> undef) #1
  ret void
}

; ALL-LABEL: 'fma_f64'
; SLOWF64: estimated cost of 4 for {{.*}} call double @llvm.fma.f64
; GFX90A-FASTF64: estimated cost of 1 for {{.*}} call double @llvm.fma.f64
; FASTF64: estimated cost of 2 for {{.*}} call double @llvm.fma.f64
; SIZEALL: estimated cost of 2 for {{.*}} call double @llvm.fma.f64
; SLOWF64: estimated cost of 8 for {{.*}} call <2 x double> @llvm.fma.v2f64
; GFX90A-FASTF64: estimated cost of 2 for {{.*}} call <2 x double> @llvm.fma.v2f64
; FASTF64: estimated cost of 4 for {{.*}} call <2 x double> @llvm.fma.v2f64
; SIZEALL: estimated cost of 4 for {{.*}} call <2 x double> @llvm.fma.v2f64
; SLOWF64: estimated cost of 12 for {{.*}} call <3 x double> @llvm.fma.v3f64
; FASTF64: estimated cost of 6 for {{.*}} call <3 x double> @llvm.fma.v3f64
; SIZEALL: estimated cost of 6 for {{.*}} call <3 x double> @llvm.fma.v3f64
define amdgpu_kernel void @fma_f64() #0 {
  %f64 = call double @llvm.fma.f64(double undef, double undef, double undef) #1
  %v2f64 = call <2 x double> @llvm.fma.v2f64(<2 x double> undef, <2 x double> undef, <2 x double> undef) #1
  %v3f64 = call <3 x double> @llvm.fma.v3f64(<3 x double> undef, <3 x double> undef, <3 x double> undef) #1
  ret void
}

; ALL-LABEL: 'fma_f16'
; SLOWF16: estimated cost of 4 for {{.*}} call half @llvm.fma.f16
; FASTF16: estimated cost of 2 for {{.*}} call half @llvm.fma.f16
; SIZEALL: estimated cost of 2 for {{.*}} call half @llvm.fma.f16
; SLOWF16: estimated cost of 8 for {{.*}} call <2 x half> @llvm.fma.v2f16
; FASTF16: estimated cost of 2 for {{.*}} call <2 x half> @llvm.fma.v2f16
; SIZEF16: estimated cost of 2 for {{.*}} call <2 x half> @llvm.fma.v2f16
; SIZENOF16: estimated cost of 4 for {{.*}} call <2 x half> @llvm.fma.v2f16
; SLOWF16: estimated cost of 16 for {{.*}} call <3 x half> @llvm.fma.v3f16
; FASTF16: estimated cost of 4 for {{.*}} call <3 x half> @llvm.fma.v3f16
; SIZEF16: estimated cost of 4 for {{.*}} call <3 x half> @llvm.fma.v3f16
; SIZENOF16: estimated cost of 8 for {{.*}} call <3 x half> @llvm.fma.v3f16
define amdgpu_kernel void @fma_f16() #0 {
  %f16 = call half @llvm.fma.f16(half undef, half undef, half undef) #1
  %v2f16 = call <2 x half> @llvm.fma.v2f16(<2 x half> undef, <2 x half> undef, <2 x half> undef) #1
  %v3f16 = call <3 x half> @llvm.fma.v3f16(<3 x half> undef, <3 x half> undef, <3 x half> undef) #1
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
