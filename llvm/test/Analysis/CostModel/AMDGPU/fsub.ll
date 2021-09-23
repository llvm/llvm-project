; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FASTF64,FASTF16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOWF64,SLOWF16,ALL %s
; RUN: opt -cost-model -analyze -cost-kind=code-size -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=SIZEALL,FASTF16,ALL %s
; RUN: opt -cost-model -analyze -cost-kind=code-size -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SIZEALL,SLOWF16,ALL %s
; END.

; ALL-LABEL: 'fsub_f32'
; ALL: estimated cost of 1 for {{.*}} fsub float
; ALL: estimated cost of 2 for {{.*}} fsub <2 x float>
; ALL: estimated cost of 3 for {{.*}} fsub <3 x float>
; ALL: estimated cost of 5 for {{.*}} fsub <5 x float>
define amdgpu_kernel void @fsub_f32() #0 {
  %f32 = fsub float undef, undef
  %v2f32 = fsub <2 x float> undef, undef
  %v3f32 = fsub <3 x float> undef, undef
  %v5f32 = fsub <5 x float> undef, undef
  ret void
}

; ALL-LABEL: 'fsub_f64'
; FASTF64: estimated cost of 2 for {{.*}} fsub double
; SLOWF64: estimated cost of 4 for {{.*}} fsub double
; SIZEALL: estimated cost of 2 for {{.*}} fsub double
; FASTF64: estimated cost of 4 for {{.*}} fsub <2 x double>
; SLOWF64: estimated cost of 8 for {{.*}} fsub <2 x double>
; SIZEALL: estimated cost of 4 for {{.*}} fsub <2 x double>
; FASTF64: estimated cost of 6 for {{.*}} fsub <3 x double>
; SLOWF64: estimated cost of 12 for {{.*}} fsub <3 x double>
; SIZEALL: estimated cost of 6 for {{.*}} fsub <3 x double>
define amdgpu_kernel void @fsub_f64() #0 {
  %f64 = fsub double undef, undef
  %v2f64 = fsub <2 x double> undef, undef
  %v3f64 = fsub <3 x double> undef, undef
  ret void
}

; ALL-LABEL: 'fsub_f16'
; ALL: estimated cost of 1 for {{.*}} fsub half
; SLOWF16: estimated cost of 2 for {{.*}} fsub <2 x half>
; FASTF16: estimated cost of 1 for {{.*}} fsub <2 x half>
; SLOWF16: estimated cost of 4 for {{.*}} fsub <3 x half>
; FASTF16: estimated cost of 2 for {{.*}} fsub <3 x half>
; SLOWF16: estimated cost of 4 for {{.*}} fsub <4 x half>
; FASTF16: estimated cost of 2 for {{.*}} fsub <4 x half>
define amdgpu_kernel void @fsub_f16() #0 {
  %f16 = fsub half undef, undef
  %v2f16 = fsub <2 x half> undef, undef
  %v3f16 = fsub <3 x half> undef, undef
  %v4f16 = fsub <4 x half> undef, undef
  ret void
}
