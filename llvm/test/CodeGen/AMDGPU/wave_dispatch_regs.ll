; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -enable-var-scope %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -enable-var-scope %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s

; This compute shader has input args that claim that it has 17 sgprs and 5 vgprs
; in wave dispatch. Ensure that the sgpr and vgpr counts in COMPUTE_PGM_RSRC1
; are set to reflect that, even though the registers are not used in the shader.

; GCN-LABEL: {{^}}_amdgpu_cs_main:
; SI: .amd_amdgpu_pal_metadata{{.*}}0x2e12,0x{{[0-9a-f]*}}81,
; VI: .amd_amdgpu_pal_metadata{{.*}}0x2e12,0x{{[0-9a-f]*}}c1,
; GFX9: .amd_amdgpu_pal_metadata{{.*}}0x2e12,0x{{[0-9a-f]*}}81,

define dllexport amdgpu_cs void @_amdgpu_cs_main(i32 inreg, i32 inreg, <2 x i32> inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, <3 x i32> inreg, i32 inreg, <5 x i32>) {
.entry:
  ret void
}

