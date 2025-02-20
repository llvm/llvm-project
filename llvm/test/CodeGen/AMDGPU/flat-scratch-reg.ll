; RUN: llc < %s -mtriple=amdgcn -mcpu=kaveri | FileCheck -check-prefix=CI -check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=fiji -mattr=-xnack | FileCheck -check-prefix=VI-NOXNACK -check-prefix=GCN %s

; RUN: llc < %s -mtriple=amdgcn -mcpu=carrizo -mattr=-xnack | FileCheck -check-prefixes=VI-NOXNACK,GCN %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=stoney -mattr=-xnack | FileCheck -check-prefixes=VI-NOXNACK,GCN %s

; RUN: llc < %s -mtriple=amdgcn -mcpu=carrizo -mattr=+xnack | FileCheck -check-prefix=VI-XNACK  -check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=stoney -mattr=+xnack | FileCheck -check-prefix=VI-XNACK  -check-prefix=GCN %s

; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck -check-prefixes=GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=-xnack | FileCheck -check-prefixes=VI-NOXNACK,HSA-VI-NOXNACK,GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=+xnack | FileCheck -check-prefixes=VI-XNACK,HSA-VI-XNACK,GCN %s

; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx900 -mattr=+architected-flat-scratch | FileCheck -check-prefixes=GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx900 -mattr=+architected-flat-scratch,-xnack | FileCheck -check-prefixes=HSA-VI-NOXNACK,GFX9-ARCH-FLAT,GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx900 -mattr=+architected-flat-scratch,+xnack | FileCheck -check-prefixes=HSA-VI-XNACK,GFX9-ARCH-FLAT,GCN %s

; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr=+architected-flat-scratch | FileCheck -check-prefixes=GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr=+architected-flat-scratch,-xnack | FileCheck -check-prefixes=HSA-VI-NOXNACK,GFX10-ARCH-FLAT,GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr=+architected-flat-scratch,+xnack | FileCheck -check-prefixes=HSA-VI-XNACK,GFX10-ARCH-FLAT,GCN %s

; GCN-LABEL: {{^}}no_vcc_no_flat:

; NOT-HSA-CI: .amdhsa_reserve_xnack_mask
; HSA-VI-NOXNACK: .amdhsa_reserve_xnack_mask 0
; HSA-VI-XNACK: .amdhsa_reserve_xnack_mask 1

; CI: ; TotalNumSgprs: 8
; VI-NOXNACK: ; TotalNumSgprs: 8
; VI-XNACK: ; TotalNumSgprs: 12
; GFX9-ARCH-FLAT: ; TotalNumSgprs: 14
; GFX10-ARCH-FLAT: ; TotalNumSgprs: 8
define amdgpu_kernel void @no_vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{s7}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_no_flat:

; NOT-HSA-CI: .amdhsa_reserve_xnack_mask
; HSA-VI-NOXNACK: .amdhsa_reserve_xnack_mask 0
; HSA-VI-XNACK: .amdhsa_reserve_xnack_mask 1

; CI: ; TotalNumSgprs: 10
; VI-NOXNACK: ; TotalNumSgprs: 10
; VI-XNACK: ; TotalNumSgprs: 12
; GFX9-ARCH-FLAT: ; TotalNumSgprs: 14
; GFX10-ARCH-FLAT: ; TotalNumSgprs: 10
define amdgpu_kernel void @vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{s7},~{vcc}"()
  ret void
}

; GCN-LABEL: {{^}}no_vcc_flat:

; NOT-HSA-CI: .amdhsa_reserve_xnack_mask
; HSA-VI-NOXNACK: .amdhsa_reserve_xnack_mask 0
; HSA-VI-XNACK: .amdhsa_reserve_xnack_mask 1

; CI: ; TotalNumSgprs: 12
; VI-NOXNACK: ; TotalNumSgprs: 14
; VI-XNACK: ; TotalNumSgprs: 14
; GFX9-ARCH-FLAT: ; TotalNumSgprs: 14
; GFX10-ARCH-FLAT: ; TotalNumSgprs: 8
define amdgpu_kernel void @no_vcc_flat() {
entry:
  call void asm sideeffect "", "~{s7},~{flat_scratch}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_flat:

; NOT-HSA-CI: .amdhsa_reserve_xnack_mask
; HSA-VI-NOXNACK: .amdhsa_reserve_xnack_mask 0
; HSA-VI-XNACK: .amdhsa_reserve_xnack_mask 1

; CI: ; TotalNumSgprs: 12
; VI-NOXNACK: ; TotalNumSgprs: 14
; VI-XNACK: ; TotalNumSgprs: 14
; GFX9-ARCH-FLAT: ; TotalNumSgprs: 14
; GFX10-ARCH-FLAT: ; TotalNumSgprs: 10
define amdgpu_kernel void @vcc_flat() {
entry:
  call void asm sideeffect "", "~{s7},~{vcc},~{flat_scratch}"()
  ret void
}

; Make sure used SGPR count for flat_scr is correct when there is no
; scratch usage and implicit flat uses.

; GCN-LABEL: {{^}}use_flat_scr:

; NOT-HSA-CI: .amdhsa_reserve_xnack_mask
; HSA-VI-NOXNACK: .amdhsa_reserve_xnack_mask 0
; HSA-VI-XNACK: .amdhsa_reserve_xnack_mask 1

; CI: NumSgprs: 4
; VI-NOXNACK: NumSgprs: 6
; VI-XNACK: NumSgprs: 6
; GFX9-ARCH-FLAT: ; TotalNumSgprs: 6
; GFX10-ARCH-FLAT: ; TotalNumSgprs: 0
define amdgpu_kernel void @use_flat_scr() #0 {
entry:
  call void asm sideeffect "; clobber ", "~{flat_scratch}"()
  ret void
}

; GCN-LABEL: {{^}}use_flat_scr_lo:

; NOT-HSA-CI: .amdhsa_reserve_xnack_mask
; HSA-VI-NOXNACK: .amdhsa_reserve_xnack_mask 0
; HSA-VI-XNACK: .amdhsa_reserve_xnack_mask 1

; CI: NumSgprs: 4
; VI-NOXNACK: NumSgprs: 6
; VI-XNACK: NumSgprs: 6
; GFX9-ARCH-FLAT: ; TotalNumSgprs: 6
; GFX10-ARCH-FLAT: ; TotalNumSgprs: 0
define amdgpu_kernel void @use_flat_scr_lo() #0 {
entry:
  call void asm sideeffect "; clobber ", "~{flat_scratch_lo}"()
  ret void
}

; GCN-LABEL: {{^}}use_flat_scr_hi:

; NOT-HSA-CI: .amdhsa_reserve_xnack_mask
; HSA-VI-NOXNACK: .amdhsa_reserve_xnack_mask 0
; HSA-VI-XNACK: .amdhsa_reserve_xnack_mask 1

; CI: NumSgprs: 4
; VI-NOXNACK: NumSgprs: 6
; VI-XNACK: NumSgprs: 6
; GFX9-ARCH-FLAT: ; TotalNumSgprs: 6
; GFX10-ARCH-FLAT: ; TotalNumSgprs: 0
define amdgpu_kernel void @use_flat_scr_hi() #0 {
entry:
  call void asm sideeffect "; clobber ", "~{flat_scratch_hi}"()
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
