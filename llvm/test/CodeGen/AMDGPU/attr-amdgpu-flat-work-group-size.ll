; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=HSAMD %s

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
; CHECK: SGPRBlocks: 8
; CHECK: VGPRBlocks: 7
; CHECK: NumSGPRsForWavesPerEU: 65
; CHECK: NumVGPRsForWavesPerEU: 29
define amdgpu_kernel void @min_128_max_128() #2 {
entry:
  ret void
}
attributes #2 = {"amdgpu-flat-work-group-size"="128,128"}

; CHECK-LABEL: {{^}}min_1024_max_1024
; CHECK: SGPRBlocks: 8
; CHECK: VGPRBlocks: 10
; CHECK: NumSGPRsForWavesPerEU: 65
; CHECK: NumVGPRsForWavesPerEU: 43
@var = addrspace(1) global float 0.0
define amdgpu_kernel void @min_1024_max_1024() #3 {
  %val0 = load volatile float, ptr addrspace(1) @var
  %val1 = load volatile float, ptr addrspace(1) @var
  %val2 = load volatile float, ptr addrspace(1) @var
  %val3 = load volatile float, ptr addrspace(1) @var
  %val4 = load volatile float, ptr addrspace(1) @var
  %val5 = load volatile float, ptr addrspace(1) @var
  %val6 = load volatile float, ptr addrspace(1) @var
  %val7 = load volatile float, ptr addrspace(1) @var
  %val8 = load volatile float, ptr addrspace(1) @var
  %val9 = load volatile float, ptr addrspace(1) @var
  %val10 = load volatile float, ptr addrspace(1) @var
  %val11 = load volatile float, ptr addrspace(1) @var
  %val12 = load volatile float, ptr addrspace(1) @var
  %val13 = load volatile float, ptr addrspace(1) @var
  %val14 = load volatile float, ptr addrspace(1) @var
  %val15 = load volatile float, ptr addrspace(1) @var
  %val16 = load volatile float, ptr addrspace(1) @var
  %val17 = load volatile float, ptr addrspace(1) @var
  %val18 = load volatile float, ptr addrspace(1) @var
  %val19 = load volatile float, ptr addrspace(1) @var
  %val20 = load volatile float, ptr addrspace(1) @var
  %val21 = load volatile float, ptr addrspace(1) @var
  %val22 = load volatile float, ptr addrspace(1) @var
  %val23 = load volatile float, ptr addrspace(1) @var
  %val24 = load volatile float, ptr addrspace(1) @var
  %val25 = load volatile float, ptr addrspace(1) @var
  %val26 = load volatile float, ptr addrspace(1) @var
  %val27 = load volatile float, ptr addrspace(1) @var
  %val28 = load volatile float, ptr addrspace(1) @var
  %val29 = load volatile float, ptr addrspace(1) @var
  %val30 = load volatile float, ptr addrspace(1) @var
  %val31 = load volatile float, ptr addrspace(1) @var
  %val32 = load volatile float, ptr addrspace(1) @var
  %val33 = load volatile float, ptr addrspace(1) @var
  %val34 = load volatile float, ptr addrspace(1) @var
  %val35 = load volatile float, ptr addrspace(1) @var
  %val36 = load volatile float, ptr addrspace(1) @var
  %val37 = load volatile float, ptr addrspace(1) @var
  %val38 = load volatile float, ptr addrspace(1) @var
  %val39 = load volatile float, ptr addrspace(1) @var
  %val40 = load volatile float, ptr addrspace(1) @var

  store volatile float %val0, ptr addrspace(1) @var
  store volatile float %val1, ptr addrspace(1) @var
  store volatile float %val2, ptr addrspace(1) @var
  store volatile float %val3, ptr addrspace(1) @var
  store volatile float %val4, ptr addrspace(1) @var
  store volatile float %val5, ptr addrspace(1) @var
  store volatile float %val6, ptr addrspace(1) @var
  store volatile float %val7, ptr addrspace(1) @var
  store volatile float %val8, ptr addrspace(1) @var
  store volatile float %val9, ptr addrspace(1) @var
  store volatile float %val10, ptr addrspace(1) @var
  store volatile float %val11, ptr addrspace(1) @var
  store volatile float %val12, ptr addrspace(1) @var
  store volatile float %val13, ptr addrspace(1) @var
  store volatile float %val14, ptr addrspace(1) @var
  store volatile float %val15, ptr addrspace(1) @var
  store volatile float %val16, ptr addrspace(1) @var
  store volatile float %val17, ptr addrspace(1) @var
  store volatile float %val18, ptr addrspace(1) @var
  store volatile float %val19, ptr addrspace(1) @var
  store volatile float %val20, ptr addrspace(1) @var
  store volatile float %val21, ptr addrspace(1) @var
  store volatile float %val22, ptr addrspace(1) @var
  store volatile float %val23, ptr addrspace(1) @var
  store volatile float %val24, ptr addrspace(1) @var
  store volatile float %val25, ptr addrspace(1) @var
  store volatile float %val26, ptr addrspace(1) @var
  store volatile float %val27, ptr addrspace(1) @var
  store volatile float %val28, ptr addrspace(1) @var
  store volatile float %val29, ptr addrspace(1) @var
  store volatile float %val30, ptr addrspace(1) @var
  store volatile float %val31, ptr addrspace(1) @var
  store volatile float %val32, ptr addrspace(1) @var
  store volatile float %val33, ptr addrspace(1) @var
  store volatile float %val34, ptr addrspace(1) @var
  store volatile float %val35, ptr addrspace(1) @var
  store volatile float %val36, ptr addrspace(1) @var
  store volatile float %val37, ptr addrspace(1) @var
  store volatile float %val38, ptr addrspace(1) @var
  store volatile float %val39, ptr addrspace(1) @var
  store volatile float %val40, ptr addrspace(1) @var

  ret void
}
attributes #3 = {"amdgpu-flat-work-group-size"="1024,1024"}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}

; HSAMD: amdhsa.kernels
; HSAMD:  .max_flat_workgroup_size: 64
; HSAMD:  .name: min_64_max_64
; HSAMD:  .max_flat_workgroup_size: 128
; HSAMD:  .name: min_64_max_128
; HSAMD:  .max_flat_workgroup_size: 128
; HSAMD:  .name: min_128_max_128
; HSAMD:  .max_flat_workgroup_size: 1024
; HSAMD:  .name: min_1024_max_1024
