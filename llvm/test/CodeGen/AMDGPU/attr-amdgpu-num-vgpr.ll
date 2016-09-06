; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

@var = addrspace(1) global float 0.0

; CHECK-LABEL: {{^}}max_20_vgprs:
; CHECK: VGPRBlocks: 4
; CHECK: NumVGPRsForWavesPerEU: 20
define void @max_20_vgprs() #1 {
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
attributes #1 = {"amdgpu-num-vgpr"="20"}
