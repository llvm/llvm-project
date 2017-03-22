; RUN: llc -march=amdgcn -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=VI-NOXNACK -check-prefix=GCN %s

; RUN: llc -march=amdgcn -mcpu=carrizo -mattr=-xnack -verify-machineinstrs < %s | FileCheck -check-prefix=VI-NOXNACK  -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=stoney -mattr=-xnack -verify-machineinstrs < %s | FileCheck -check-prefix=VI-NOXNACK  -check-prefix=GCN %s

; RUN: llc -march=amdgcn -mcpu=carrizo -verify-machineinstrs < %s | FileCheck -check-prefix=VI-XNACK  -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=stoney  -verify-machineinstrs < %s | FileCheck -check-prefix=VI-XNACK  -check-prefix=GCN %s

; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=HSA-CI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=-xnack -verify-machineinstrs < %s | FileCheck -check-prefix=HSA-VI-NOXNACK -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=+xnack -verify-machineinstrs < %s | FileCheck -check-prefix=HSA-VI-XNACK -check-prefix=GCN %s

; GCN-LABEL: {{^}}no_vcc_no_flat:
; HSA-CI: is_xnack_enabled = 0
; HSA-VI-NOXNACK: is_xnack_enabled = 0
; HSA-VI-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 8
; VI-NOXNACK: ; NumSgprs: 8
; VI-XNACK: ; NumSgprs: 12
define amdgpu_kernel void @no_vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7}"()
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
  call void asm sideeffect "", "~{SGPR7},~{VCC}"()
  ret void
}

; GCN-LABEL: {{^}}no_vcc_flat:
; HSA-CI: is_xnack_enabled = 0
; HSA-VI-NOXNACK: is_xnack_enabled = 0
; HSA-VI-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 8
; VI-NOXNACK: ; NumSgprs: 8
; VI-XNACK: ; NumSgprs: 12
; HSA-CI: ; NumSgprs: 8
; HSA-VI-NOXNACK: ; NumSgprs: 8
; HSA-VI-XNACK: ; NumSgprs: 12
define amdgpu_kernel void @no_vcc_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7},~{FLAT_SCR}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_flat:
; HSA-NOXNACK: is_xnack_enabled = 0
; HSA-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 10
; VI-NOXNACK: ; NumSgprs: 10
; VI-XNACK: ; NumSgprs: 12
; HSA-CI: ; NumSgprs: 10
; HSA-VI-NOXNACK: ; NumSgprs: 10
; HSA-VI-XNACK: ; NumSgprs: 12
define amdgpu_kernel void @vcc_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7},~{VCC},~{FLAT_SCR}"()
  ret void
}
