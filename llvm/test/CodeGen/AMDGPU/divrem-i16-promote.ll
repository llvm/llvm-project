; RUN: llc -amdgpu-codegenprepare-idiv-expansion=0 -amdgpu-bypass-slow-div=0 -mtriple=amdgpu8.03-amd-amdhsa < %s | FileCheck %s
; RUN: llc -amdgpu-codegenprepare-idiv-expansion=0 -amdgpu-bypass-slow-div=0 -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck %s
; RUN: llc -amdgpu-codegenprepare-idiv-expansion=0 -amdgpu-bypass-slow-div=0 -mtriple=amdgpu10.10-amd-amdhsa < %s | FileCheck %s

; Make sure that div/rem pairs on i8/i16 (promoted to i16 on targets with
; 16-bit instructions) are not combined into an unselectable i16 [SU]DIVREM.

; CHECK-LABEL: {{^}}sdivrem_i8:
; CHECK: s_endpgm
define amdgpu_kernel void @sdivrem_i8(ptr addrspace(1) %out0, ptr addrspace(1) %out1, i8 %x, i8 %y) {
  %div = sdiv i8 %x, %y
  store i8 %div, ptr addrspace(1) %out0
  %rem = srem i8 %x, %y
  store i8 %rem, ptr addrspace(1) %out1
  ret void
}

; CHECK-LABEL: {{^}}udivrem_i8:
; CHECK: s_endpgm
define amdgpu_kernel void @udivrem_i8(ptr addrspace(1) %out0, ptr addrspace(1) %out1, i8 %x, i8 %y) {
  %div = udiv i8 %x, %y
  store i8 %div, ptr addrspace(1) %out0
  %rem = urem i8 %x, %y
  store i8 %rem, ptr addrspace(1) %out1
  ret void
}

; CHECK-LABEL: {{^}}sdivrem_i16:
; CHECK: s_endpgm
define amdgpu_kernel void @sdivrem_i16(ptr addrspace(1) %out0, ptr addrspace(1) %out1, i16 %x, i16 %y) {
  %div = sdiv i16 %x, %y
  store i16 %div, ptr addrspace(1) %out0
  %rem = srem i16 %x, %y
  store i16 %rem, ptr addrspace(1) %out1
  ret void
}

; CHECK-LABEL: {{^}}udivrem_i16:
; CHECK: s_endpgm
define amdgpu_kernel void @udivrem_i16(ptr addrspace(1) %out0, ptr addrspace(1) %out1, i16 %x, i16 %y) {
  %div = udiv i16 %x, %y
  store i16 %div, ptr addrspace(1) %out0
  %rem = urem i16 %x, %y
  store i16 %rem, ptr addrspace(1) %out1
  ret void
}
