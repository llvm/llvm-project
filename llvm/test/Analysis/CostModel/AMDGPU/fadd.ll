; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900  -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FASTF64,FASTF16,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOWF64,SLOWF16,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900  -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=FASTF16,SIZEALL,ALL %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefixes=SLOWF16,SIZEALL,ALL %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx90a  -mattr=+half-rate-64-ops < %s | FileCheck -check-prefixes=GFX90A-FASTF64,FASTF16,PACKEDF32,ALL %s
; END.

; ALL-LABEL: 'fadd_f32'
; ALL: estimated cost of 1 for {{.*}} fadd float
; NOPACKEDF32: estimated cost of 2 for {{.*}} fadd <2 x float>
; PACKEDF32: estimated cost of 1 for {{.*}} fadd <2 x float>
; Allow for 4 when v3f32 is illegal and TargetLowering thinks it needs widening,
; and 3 when it is legal.
; NOPACKEDF32: estimated cost of {{[34]}} for {{.*}} fadd <3 x float>
; PACKEDF32: estimated cost of 2 for {{.*}} fadd <3 x float>
; Allow for 8 when v5f32 is illegal and TargetLowering thinks it needs widening,
; and 5 when it is legal.
; NOPACKEDF32: estimated cost of {{[58]}} for {{.*}} fadd <5 x float>
; PACKEDF32: estimated cost of 3 for {{.*}} fadd <5 x float>
define amdgpu_kernel void @fadd_f32() #0 {
  %f32 = fadd float undef, undef
  %v2f32 = fadd <2 x float> undef, undef
  %v3f32 = fadd <3 x float> undef, undef
  %v5f32 = fadd <5 x float> undef, undef
  ret void
}

; ALL-LABEL: 'fadd_f64'
; GFX90A-FASTF64: estimated cost of 1 for {{.*}} fadd double
; FASTF64: estimated cost of 2 for {{.*}} fadd double
; SLOWF64: estimated cost of 4 for {{.*}} fadd double
; SIZEALL: estimated cost of 2 for {{.*}} fadd double
; GFX90A-FASTF64: estimated cost of 2 for {{.*}} fadd <2 x double>
; FASTF64: estimated cost of 4 for {{.*}} fadd <2 x double>
; SLOWF64: estimated cost of 8 for {{.*}} fadd <2 x double>
; SIZEALL: estimated cost of 4 for {{.*}} fadd <2 x double>
; GFX90A-FASTF64: estimated cost of 3 for {{.*}} fadd <3 x double>
; FASTF64: estimated cost of 6 for {{.*}} fadd <3 x double>
; SLOWF64: estimated cost of 12 for {{.*}} fadd <3 x double>
; SIZEALL: estimated cost of 6 for {{.*}} fadd <3 x double>
define amdgpu_kernel void @fadd_f64() #0 {
  %f64 = fadd double undef, undef
  %v2f64 = fadd <2 x double> undef, undef
  %v3f64 = fadd <3 x double> undef, undef
  ret void
}

; ALL-LABEL: 'fadd_f16'
; ALL: estimated cost of 1 for {{.*}} fadd half
; SLOWF16: estimated cost of 2 for {{.*}} fadd <2 x half>
; FASTF16: estimated cost of 1 for {{.*}} fadd <2 x half>
; SLOWF16: estimated cost of 4 for {{.*}} fadd <3 x half>
; FASTF16: estimated cost of 2 for {{.*}} fadd <3 x half>
; SLOWF16: estimated cost of 4 for {{.*}} fadd <4 x half>
; FASTF16: estimated cost of 2 for {{.*}} fadd <4 x half>
define amdgpu_kernel void @fadd_f16() #0 {
  %f16 = fadd half undef, undef
  %v2f16 = fadd <2 x half> undef, undef
  %v3f16 = fadd <3 x half> undef, undef
  %v4f16 = fadd <4 x half> undef, undef
  ret void
}

attributes #0 = { nounwind }
