; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 --amdhsa-code-object-version=3 -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 --amdhsa-code-object-version=3 -verify-machineinstrs -amdgpu-verify-hsa-metadata -filetype=obj -o /dev/null < %s 2>&1 | FileCheck --check-prefix=PARSER %s

; CHECK-LABEL: {{^}}min_64_max_64:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @min_64_max_64() #0 {
entry:
  ret void
}
attributes #0 = {"amdgpu-flat-work-group-size"="64,64"}

; CHECK-LABEL: {{^}}min_64_max_128:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @min_64_max_128() #1 {
entry:
  ret void
}
attributes #1 = {"amdgpu-flat-work-group-size"="64,128"}

; CHECK-LABEL: {{^}}min_128_max_128:
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 0
; CHECK: NumSGPRsForWavesPerEU: 1
; CHECK: NumVGPRsForWavesPerEU: 1
define amdgpu_kernel void @min_128_max_128() #2 {
entry:
  ret void
}
attributes #2 = {"amdgpu-flat-work-group-size"="128,128"}

; CHECK-LABEL: {{^}}min_1024_max_1024
; CHECK: SGPRBlocks: 0
; CHECK: VGPRBlocks: 10
; CHECK: NumSGPRsForWavesPerEU: 2{{$}}
; CHECK: NumVGPRsForWavesPerEU: 43
@var = addrspace(1) global float 0.0
define amdgpu_kernel void @min_1024_max_1024() #3 {
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
  %val31 = load volatile float, float addrspace(1)* @var
  %val32 = load volatile float, float addrspace(1)* @var
  %val33 = load volatile float, float addrspace(1)* @var
  %val34 = load volatile float, float addrspace(1)* @var
  %val35 = load volatile float, float addrspace(1)* @var
  %val36 = load volatile float, float addrspace(1)* @var
  %val37 = load volatile float, float addrspace(1)* @var
  %val38 = load volatile float, float addrspace(1)* @var
  %val39 = load volatile float, float addrspace(1)* @var
  %val40 = load volatile float, float addrspace(1)* @var

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
  store volatile float %val31, float addrspace(1)* @var
  store volatile float %val32, float addrspace(1)* @var
  store volatile float %val33, float addrspace(1)* @var
  store volatile float %val34, float addrspace(1)* @var
  store volatile float %val35, float addrspace(1)* @var
  store volatile float %val36, float addrspace(1)* @var
  store volatile float %val37, float addrspace(1)* @var
  store volatile float %val38, float addrspace(1)* @var
  store volatile float %val39, float addrspace(1)* @var
  store volatile float %val40, float addrspace(1)* @var

  ret void
}
attributes #3 = {"amdgpu-flat-work-group-size"="1024,1024"}

; CHECK: amdhsa.kernels:
; CHECK:   .max_flat_workgroup_size: 64
; CHECK:   .name:                 min_64_max_64
; CHECK:   .max_flat_workgroup_size: 128
; CHECK:   .name:                 min_64_max_128
; CHECK:   .max_flat_workgroup_size: 128
; CHECK:   .name:                 min_128_max_128
; CHECK:   .max_flat_workgroup_size: 1024
; CHECK:   .name:                 min_1024_max_1024
; CHECK: amdhsa.version:
; CHECK:   - 1
; CHECK:   - 0

; PARSER: AMDGPU HSA Metadata Parser Test: PASS
