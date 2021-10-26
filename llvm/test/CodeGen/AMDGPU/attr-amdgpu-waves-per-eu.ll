; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

; Exactly 1 wave per execution unit.
; CHECK-LABEL: {{^}}empty_exactly_1:
; CHECK: SGPRBlocks: 12
; CHECK: VGPRBlocks: 32
; CHECK: NumSGPRsForWavesPerEU: 102
; CHECK: NumVGPRsForWavesPerEU: 129
define amdgpu_kernel void @empty_exactly_1() #0 {
entry:
  ret void
}
attributes #0 = {"amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,64" }

; Exactly 5 waves per execution unit.
; CHECK-LABEL: {{^}}empty_exactly_5:
; CHECK: SGPRBlocks: 12
; CHECK: VGPRBlocks: 10
; CHECK: NumSGPRsForWavesPerEU: 102
; CHECK: NumVGPRsForWavesPerEU: 41
define amdgpu_kernel void @empty_exactly_5() #1 {
entry:
  ret void
}
attributes #1 = {"amdgpu-waves-per-eu"="5,5"}

; Exactly 10 waves per execution unit.
; CHECK-LABEL: {{^}}empty_exactly_10:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @empty_exactly_10() #2 {
entry:
  ret void
}
attributes #2 = {"amdgpu-waves-per-eu"="10,10"}

; At least 1 wave per execution unit.
; CHECK-LABEL: {{^}}empty_at_least_1:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @empty_at_least_1() #3 {
entry:
  ret void
}
attributes #3 = {"amdgpu-waves-per-eu"="1"}

; At least 5 waves per execution unit.
; CHECK-LABEL: {{^}}empty_at_least_5:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @empty_at_least_5() #4 {
entry:
  ret void
}
attributes #4 = {"amdgpu-waves-per-eu"="5"}

; At least 10 waves per execution unit.
; CHECK-LABEL: {{^}}empty_at_least_10:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @empty_at_least_10() #5 {
entry:
  ret void
}
attributes #5 = {"amdgpu-waves-per-eu"="10"}

; At most 1 wave per execution unit (same as @empty_exactly_1).

; At most 5 waves per execution unit.
; CHECK-LABEL: {{^}}empty_at_most_5:
; CHECK: SGPRBlocks: 12
; CHECK: VGPRBlocks: 10
; CHECK: NumSGPRsForWavesPerEU: 102
; CHECK: NumVGPRsForWavesPerEU: 41
define amdgpu_kernel void @empty_at_most_5() #6 {
entry:
  ret void
}
attributes #6 = {"amdgpu-waves-per-eu"="1,5" "amdgpu-flat-work-group-size"="1,64"}

; At most 10 waves per execution unit.
; CHECK-LABEL: {{^}}empty_at_most_10:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @empty_at_most_10() #7 {
entry:
  ret void
}
attributes #7 = {"amdgpu-waves-per-eu"="1,10"}

; Between 1 and 5 waves per execution unit (same as @empty_at_most_5).

; Between 5 and 10 waves per execution unit.
; CHECK-LABEL: {{^}}empty_between_5_and_10:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @empty_between_5_and_10() #8 {
entry:
  ret void
}
attributes #8 = {"amdgpu-waves-per-eu"="5,10"}

@var = addrspace(1) global float 0.0

; Exactly 10 waves per execution unit.
; CHECK-LABEL: {{^}}exactly_10:
; CHECK: SGPRBlocks: 1
; CHECK: VGPRBlocks: 5
; CHECK: NumSGPRsForWavesPerEU: 12
; CHECK: NumVGPRsForWavesPerEU: 24
define amdgpu_kernel void @exactly_10() #9 {
  %val0 = load volatile float, float addrspace(1)* @var
  %val1 = load volatile float, float addrspace(1)* @var
  %val2 = load volatile float, float addrspace(1)* @var
  %val3 = load volatile float, float addrspace(1)* @var
  %val4 = load volatile float, float addrspace(1)* @var
  %val5 = load volatile float, float addrspace(1)* @var
  %val6 = load volatile float, float addrspace(1)* @var
  %val7 = load volatile float, float addrspace(1)* @var
  %val8 = load volatile float, float addrspace(1)* @var
  %val9 = load volatile float, float addrspace(1)* @var
  %val10 = load volatile float, float addrspace(1)* @var
  %val11 = load volatile float, float addrspace(1)* @var
  %val12 = load volatile float, float addrspace(1)* @var
  %val13 = load volatile float, float addrspace(1)* @var
  %val14 = load volatile float, float addrspace(1)* @var
  %val15 = load volatile float, float addrspace(1)* @var
  %val16 = load volatile float, float addrspace(1)* @var
  %val17 = load volatile float, float addrspace(1)* @var
  %val18 = load volatile float, float addrspace(1)* @var
  %val19 = load volatile float, float addrspace(1)* @var
  %val20 = load volatile float, float addrspace(1)* @var
  %val21 = load volatile float, float addrspace(1)* @var
  %val22 = load volatile float, float addrspace(1)* @var
  %val23 = load volatile float, float addrspace(1)* @var
  %val24 = load volatile float, float addrspace(1)* @var
  %val25 = load volatile float, float addrspace(1)* @var
  %val26 = load volatile float, float addrspace(1)* @var
  %val27 = load volatile float, float addrspace(1)* @var
  %val28 = load volatile float, float addrspace(1)* @var
  %val29 = load volatile float, float addrspace(1)* @var
  %val30 = load volatile float, float addrspace(1)* @var

  store volatile float %val0, float addrspace(1)* @var
  store volatile float %val1, float addrspace(1)* @var
  store volatile float %val2, float addrspace(1)* @var
  store volatile float %val3, float addrspace(1)* @var
  store volatile float %val4, float addrspace(1)* @var
  store volatile float %val5, float addrspace(1)* @var
  store volatile float %val6, float addrspace(1)* @var
  store volatile float %val7, float addrspace(1)* @var
  store volatile float %val8, float addrspace(1)* @var
  store volatile float %val9, float addrspace(1)* @var
  store volatile float %val10, float addrspace(1)* @var
  store volatile float %val11, float addrspace(1)* @var
  store volatile float %val12, float addrspace(1)* @var
  store volatile float %val13, float addrspace(1)* @var
  store volatile float %val14, float addrspace(1)* @var
  store volatile float %val15, float addrspace(1)* @var
  store volatile float %val16, float addrspace(1)* @var
  store volatile float %val17, float addrspace(1)* @var
  store volatile float %val18, float addrspace(1)* @var
  store volatile float %val19, float addrspace(1)* @var
  store volatile float %val20, float addrspace(1)* @var
  store volatile float %val21, float addrspace(1)* @var
  store volatile float %val22, float addrspace(1)* @var
  store volatile float %val23, float addrspace(1)* @var
  store volatile float %val24, float addrspace(1)* @var
  store volatile float %val25, float addrspace(1)* @var
  store volatile float %val26, float addrspace(1)* @var
  store volatile float %val27, float addrspace(1)* @var
  store volatile float %val28, float addrspace(1)* @var
  store volatile float %val29, float addrspace(1)* @var
  store volatile float %val30, float addrspace(1)* @var

  ret void
}
attributes #9 = {"amdgpu-waves-per-eu"="10,10"}

; Exactly 256 workitems and exactly 2 waves.
; CHECK-LABEL: {{^}}empty_workitems_exactly_256_waves_exactly_2:
; CHECK: SGPRBlocks: 12
; CHECK: VGPRBlocks: 21
; CHECK: NumSGPRsForWavesPerEU: 102
; CHECK: NumVGPRsForWavesPerEU: 85
define amdgpu_kernel void @empty_workitems_exactly_256_waves_exactly_2() #10 {
entry:
  ret void
}
attributes #10 = {"amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="2,2"}
