; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck -check-prefixes=GCN,GFX600 %s
; RUN: llc -mtriple=amdgpu8.03 < %s | FileCheck -check-prefixes=GCN,GFX803 %s
; RUN: llc -mtriple=amdgpu13.10 < %s | FileCheck -check-prefixes=GCN,GFX13 %s

declare i32 @llvm.amdgcn.msad.u8(i32, i32, i32) #0

; GCN-LABEL: {{^}}v_msad_u8:
; GFX600: v_msad_u8 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GFX803: v_msad_u8 v{{[0-9]+}}, v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GFX13: v_msad_u8 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @v_msad_u8(ptr addrspace(1) %out, i32 %src) {
  %result= call i32 @llvm.amdgcn.msad.u8(i32 %src, i32 100, i32 100) #0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_msad_u8_non_immediate:
; GFX600: v_msad_u8 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX803: v_msad_u8 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX13: v_msad_u8 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_msad_u8_non_immediate(ptr addrspace(1) %out, i32 %src, i32 %a, i32 %b) {
  %result= call i32 @llvm.amdgcn.msad.u8(i32 %src, i32 %a, i32 %b) #0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
