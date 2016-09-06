; RUN: llc -march=amdgcn -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=NOXNACK -check-prefix=CI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=NOXNACK -check-prefix=VI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=carrizo -mattr=+xnack -verify-machineinstrs < %s | FileCheck -check-prefix=XNACK -check-prefix=VI  -check-prefix=GCN %s

; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=NOXNACK -check-prefix=HSA-NOXNACK -check-prefix=HSA -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=-xnack -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=NOXNACK -check-prefix=HSA-NOXNACK -check-prefix=HSA -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=+xnack -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=XNACK -check-prefix=HSA-XNACK -check-prefix=HSA -check-prefix=GCN %s

; GCN-LABEL: {{^}}no_vcc_no_flat:
; HSA-NOXNACK: is_xnack_enabled = 0
; HSA-XNACK: is_xnack_enabled = 1

; NOXNACK: ; NumSgprs: 8
; XNACK: ; NumSgprs: 12
define void @no_vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_no_flat:
; HSA-NOXNACK: is_xnack_enabled = 0
; HSA-XNACK: is_xnack_enabled = 1

; NOXNACK: ; NumSgprs: 10
; XNACK: ; NumSgprs: 12
define void @vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7},~{VCC}"()
  ret void
}

; GCN-LABEL: {{^}}no_vcc_flat:
; HSA-NOXNACK: is_xnack_enabled = 0
; HSA-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 12
; VI: ; NumSgprs: 14
define void @no_vcc_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7},~{FLAT_SCR}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_flat:
; HSA-NOXNACK: is_xnack_enabled = 0
; HSA-XNACK: is_xnack_enabled = 1

; CI: ; NumSgprs: 12
; VI: ; NumSgprs: 14
define void @vcc_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7},~{VCC},~{FLAT_SCR}"()
  ret void
}
