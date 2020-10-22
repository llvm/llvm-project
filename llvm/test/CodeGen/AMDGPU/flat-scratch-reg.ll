; RUN: llc -march=amdgcn -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=VI-NOXNACK -check-prefix=GCN %s

; RUN: llc -march=amdgcn -mcpu=carrizo -mattr=-xnack -verify-machineinstrs < %s | FileCheck -check-prefix=VI-NOXNACK  -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=stoney -mattr=-xnack -verify-machineinstrs < %s | FileCheck -check-prefix=VI-NOXNACK  -check-prefix=GCN %s

; RUN: llc -march=amdgcn -mcpu=carrizo -verify-machineinstrs < %s | FileCheck -check-prefix=VI-XNACK  -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=stoney  -verify-machineinstrs < %s | FileCheck -check-prefix=VI-XNACK  -check-prefix=GCN %s

; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=kaveri --amdhsa-code-object-version=2 -verify-machineinstrs < %s | FileCheck -check-prefix=HSA-CI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=carrizo --amdhsa-code-object-version=2 -mattr=-xnack -verify-machineinstrs < %s | FileCheck -check-prefix=HSA-VI-NOXNACK -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=carrizo --amdhsa-code-object-version=2 -mattr=+xnack -verify-machineinstrs < %s | FileCheck -check-prefix=HSA-VI-XNACK -check-prefix=GCN %s

; GCN-LABEL: {{^}}no_vcc_no_flat:
; HSA-CI: is_xnack_enabled = 0
; HSA-VI-NOXNACK: is_xnack_enabled = 0
; HSA-VI-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 8
; VI-NOXNACK: ; NumSgprs: 8
; VI-XNACK: ; NumSgprs: 12
define amdgpu_kernel void @no_vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{s7}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_no_flat:
; HSA-CI: is_xnack_enabled = 0
; HSA-VI-NOXNACK: is_xnack_enabled = 0
; HSA-VI-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 10
; VI-NOXNACK: ; NumSgprs: 10
; VI-XNACK: ; NumSgprs: 12
define amdgpu_kernel void @vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{s7},~{vcc}"()
  ret void
}

; GCN-LABEL: {{^}}no_vcc_flat:
; HSA-CI: is_xnack_enabled = 0
; HSA-VI-NOXNACK: is_xnack_enabled = 0
; HSA-VI-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 12
; VI-NOXNACK: ; NumSgprs: 14
; VI-XNACK: ; NumSgprs: 14
; HSA-CI: ; NumSgprs: 12
; HSA-VI-NOXNACK: ; NumSgprs: 14
; HSA-VI-XNACK: ; NumSgprs: 14
define amdgpu_kernel void @no_vcc_flat() {
entry:
  call void asm sideeffect "", "~{s7},~{flat_scratch}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_flat:
; HSA-NOXNACK: is_xnack_enabled = 0
; HSA-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 12
; VI-NOXNACK: ; NumSgprs: 14
; VI-XNACK: ; NumSgprs: 14
; HSA-CI: ; NumSgprs: 12
; HSA-VI-NOXNACK: ; NumSgprs: 14
; HSA-VI-XNACK: ; NumSgprs: 14
define amdgpu_kernel void @vcc_flat() {
entry:
  call void asm sideeffect "", "~{s7},~{vcc},~{flat_scratch}"()
  ret void
}

; Make sure used SGPR count for flat_scr is correct when there is no
; scratch usage and implicit flat uses.

; GCN-LABEL: {{^}}use_flat_scr:
; CI: NumSgprs: 4
; VI-NOXNACK: NumSgprs: 6
; VI-XNACK: NumSgprs: 6
define amdgpu_kernel void @use_flat_scr() #0 {
entry:
  call void asm sideeffect "; clobber ", "~{flat_scratch}"()
  ret void
}

; GCN-LABEL: {{^}}use_flat_scr_lo:
; CI: NumSgprs: 4
; VI-NOXNACK: NumSgprs: 6
; VI-XNACK: NumSgprs: 6
define amdgpu_kernel void @use_flat_scr_lo() #0 {
entry:
  call void asm sideeffect "; clobber ", "~{flat_scratch_lo}"()
  ret void
}

; GCN-LABEL: {{^}}use_flat_scr_hi:
; CI: NumSgprs: 4
; VI-NOXNACK: NumSgprs: 6
; VI-XNACK: NumSgprs: 6
define amdgpu_kernel void @use_flat_scr_hi() #0 {
entry:
  call void asm sideeffect "; clobber ", "~{flat_scratch_hi}"()
  ret void
}

attributes #0 = { nounwind }
