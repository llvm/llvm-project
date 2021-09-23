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
; END.

; ALL-LABEL: 'fdiv_f32_ieee'
; THRPTALL: estimated cost of 14 for {{.*}} fdiv float
; THRPTALL: estimated cost of 28 for {{.*}} fdiv <2 x float>
; THRPTALL: estimated cost of 42 for {{.*}} fdiv <3 x float>
; THRPTALL: estimated cost of 70 for {{.*}} fdiv <5 x float>
; SIZEALL: estimated cost of 12 for {{.*}} fdiv float
; SIZEALL: estimated cost of 24 for {{.*}} fdiv <2 x float>
; SIZEALL: estimated cost of 36 for {{.*}} fdiv <3 x float>
; SIZEALL: estimated cost of 60 for {{.*}} fdiv <5 x float>
define amdgpu_kernel void @fdiv_f32_ieee() #0 {
  %f32 = fdiv float undef, undef
  %v2f32 = fdiv <2 x float> undef, undef
  %v3f32 = fdiv <3 x float> undef, undef
  %v5f32 = fdiv <5 x float> undef, undef
  ret void
}

; ALL-LABEL: 'fdiv_f32_ftzdaz'
; THRPTALL: estimated cost of 16 for {{.*}} fdiv float
; SIZEALL: estimated cost of 14 for {{.*}} fdiv float
; THRPTALL: estimated cost of 32 for {{.*}} fdiv <2 x float>
; SIZEALL: estimated cost of 28 for {{.*}} fdiv <2 x float>
; THRPTALL: estimated cost of 48 for {{.*}} fdiv <3 x float>
; SIZEALL: estimated cost of 42 for {{.*}} fdiv <3 x float>
; THRPTALL: estimated cost of 80 for {{.*}} fdiv <5 x float>
; SIZEALL: estimated cost of 70 for {{.*}} fdiv <5 x float>
define amdgpu_kernel void @fdiv_f32_ftzdaz() #1 {
  %f32 = fdiv float undef, undef
  %v2f32 = fdiv <2 x float> undef, undef
  %v3f32 = fdiv <3 x float> undef, undef
  %v5f32 = fdiv <5 x float> undef, undef
  ret void
}

; ALL-LABEL: 'fdiv_f64'
; CIFASTF64: estimated cost of 24 for {{.*}} fdiv double
; CISLOWF64: estimated cost of 38 for {{.*}} fdiv double
; SIFASTF64: estimated cost of 27 for {{.*}} fdiv double
; SISLOWF64: estimated cost of 41 for {{.*}} fdiv double
; SIZECI: estimated cost of 22 for {{.*}} fdiv double
; SIZESI: estimated cost of 25 for {{.*}} fdiv double
; CIFASTF64: estimated cost of 48 for {{.*}} fdiv <2 x double>
; CISLOWF64: estimated cost of 76 for {{.*}} fdiv <2 x double>
; SIFASTF64: estimated cost of 54 for {{.*}} fdiv <2 x double>
; SISLOWF64: estimated cost of 82 for {{.*}} fdiv <2 x double>
; SIZECI: estimated cost of 44 for {{.*}} fdiv <2 x double>
; SIZESI: estimated cost of 50 for {{.*}} fdiv <2 x double>
; CIFASTF64: estimated cost of 72 for {{.*}} fdiv <3 x double>
; CISLOWF64: estimated cost of 114 for {{.*}} fdiv <3 x double>
; SIFASTF64: estimated cost of 81 for {{.*}} fdiv <3 x double>
; SISLOWF64: estimated cost of 123 for {{.*}} fdiv <3 x double>
; SIZECI: estimated cost of 66 for {{.*}} fdiv <3 x double>
; SIZESI: estimated cost of 75 for {{.*}} fdiv <3 x double>
define amdgpu_kernel void @fdiv_f64() #0 {
  %f64 = fdiv double undef, undef
  %v2f64 = fdiv <2 x double> undef, undef
  %v3f64 = fdiv <3 x double> undef, undef
  ret void
}

; ALL-LABEL: 'fdiv_f16_f32ieee'
; NOFP16: estimated cost of 14 for {{.*}} fdiv half
; FP16: estimated cost of 12 for {{.*}} fdiv half
; SIZENOF16: estimated cost of 12 for {{.*}} fdiv half
; SIZEF16: estimated cost of 8 for {{.*}} fdiv half
; NOFP16: estimated cost of 28 for {{.*}} fdiv <2 x half>
; FP16: estimated cost of 24 for {{.*}} fdiv <2 x half>
; SIZENOF16: estimated cost of 24 for {{.*}} fdiv <2 x half>
; SIZEF16: estimated cost of 16 for {{.*}} fdiv <2 x half>
; NOFP16: estimated cost of 56 for {{.*}} fdiv <4 x half>
; FP16: estimated cost of 48 for {{.*}} fdiv <4 x half>
; SIZENOF16: estimated cost of 48 for {{.*}} fdiv <4 x half>
; SIZEF16: estimated cost of 32 for {{.*}} fdiv <4 x half>
define amdgpu_kernel void @fdiv_f16_f32ieee() #0 {
  %f16 = fdiv half undef, undef
  %v2f16 = fdiv <2 x half> undef, undef
  %v4f16 = fdiv <4 x half> undef, undef
  ret void
}

; ALL-LABEL: 'fdiv_f16_f32ftzdaz'
; NOFP16: estimated cost of 16 for {{.*}} fdiv half
; FP16: estimated cost of 12 for {{.*}} fdiv half
; SIZENOF16: estimated cost of 14 for {{.*}} fdiv half
; SIZEF16: estimated cost of 8 for {{.*}} fdiv half
; NOFP16: estimated cost of 32 for {{.*}} fdiv <2 x half>
; FP16: estimated cost of 24 for {{.*}} fdiv <2 x half>
; SIZENOF16: estimated cost of 28 for {{.*}} fdiv <2 x half>
; SIZEF16: estimated cost of 16 for {{.*}} fdiv <2 x half>
; NOFP16: estimated cost of 64 for {{.*}} fdiv <4 x half>
; FP16: estimated cost of 48 for {{.*}} fdiv <4 x half>
; SIZENOF16: estimated cost of 56 for {{.*}} fdiv <4 x half>
; SIZEF16: estimated cost of 32 for {{.*}} fdiv <4 x half>
define amdgpu_kernel void @fdiv_f16_f32ftzdaz() #1 {
  %f16 = fdiv half undef, undef
  %v2f16 = fdiv <2 x half> undef, undef
  %v4f16 = fdiv <4 x half> undef, undef
  ret void
}

; ALL-LABEL: 'rcp_ieee'
; THRPTALL: estimated cost of 14 for {{.*}} fdiv float
; SIZEALL: estimated cost of 12 for {{.*}} fdiv float
; NOFP16: estimated cost of 14 for {{.*}} fdiv half
; FP16: estimated cost of 4 for {{.*}} fdiv half
; SIZENOF16: estimated cost of 12 for {{.*}} fdiv half
; SIZEF16: estimated cost of 2 for {{.*}} fdiv half
; CIFASTF64: estimated cost of 24 for {{.*}} fdiv double
; CISLOWF64: estimated cost of 38 for {{.*}} fdiv double
; SIFASTF64: estimated cost of 27 for {{.*}} fdiv double
; SISLOWF64: estimated cost of 41 for {{.*}} fdiv double
; SIZECI: estimated cost of 22 for {{.*}} fdiv double
; SIZESI: estimated cost of 25 for {{.*}} fdiv double
; THRPTALL: estimated cost of 28 for {{.*}} fdiv <2 x float>
; SIZEALL: estimated cost of 24 for {{.*}} fdiv <2 x float>
; NOFP16: estimated cost of 28 for {{.*}} fdiv <2 x half>
; FP16: estimated cost of 8 for {{.*}} fdiv <2 x half>
; SIZENOF16: estimated cost of 24 for {{.*}} fdiv <2 x half>
; SIZEF16: estimated cost of 4 for {{.*}} fdiv <2 x half>
define amdgpu_kernel void @rcp_ieee() #0 {
  %f32 = fdiv float 1.0, undef
  %f16 = fdiv half 1.0, undef
  %f64 = fdiv double 1.0, undef
  %v2f32 = fdiv <2 x float> <float 1.0, float 1.0>, undef
  %v2f16 = fdiv <2 x half> <half 1.0, half 1.0>, undef
  ret void
}

; ALL-LABEL: 'rcp_ftzdaz'
; THRPTALL: estimated cost of 4 for {{.*}} fdiv float
; SIZEALL: estimated cost of 2 for {{.*}} fdiv float
; THRPTALL: estimated cost of 4 for {{.*}} fdiv half
; SIZEALL: estimated cost of 2 for {{.*}} fdiv half
; THRPTALL: estimated cost of 8 for {{.*}} fdiv <2 x float>
; SIZEALL: estimated cost of 4 for {{.*}} fdiv <2 x float>
; THRPTALL: estimated cost of 8 for {{.*}} fdiv <2 x half>
; SIZEALL: estimated cost of 4 for {{.*}} fdiv <2 x half>
define amdgpu_kernel void @rcp_ftzdaz() #1 {
  %f32 = fdiv float 1.0, undef
  %f16 = fdiv half 1.0, undef
  %v2f32 = fdiv <2 x float> <float 1.0, float 1.0>, undef
  %v2f16 = fdiv <2 x half> <half 1.0, half 1.0>, undef
  ret void
}

attributes #0 = { nounwind "denormal-fp-math-f32"="ieee,ieee" }
attributes #1 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
