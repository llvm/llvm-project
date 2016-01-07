; RUN: llc < %s -march=amdgcn -mcpu=kaveri -verify-machineinstrs | FileCheck %s --check-prefix=GCN --check-prefix=CI --check-prefix=NO-XNACK
; RUN: llc < %s -march=amdgcn -mcpu=fiji -verify-machineinstrs | FileCheck %s --check-prefix=GCN --check-prefix=VI --check-prefix=NO-XNACK
; RUN: llc < %s -march=amdgcn -mcpu=carrizo -mattr=+xnack -verify-machineinstrs | FileCheck %s --check-prefix=GCN --check-prefix=VI --check-prefix=XNACK

; GCN-LABEL: {{^}}no_vcc_no_flat:
; NO-XNACK: ; NumSgprs: 8
; XNACK: ; NumSgprs: 12
define void @no_vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_no_flat:
; NO-XNACK: ; NumSgprs: 10
; XNACK: ; NumSgprs: 12
define void @vcc_no_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7},~{VCC}"()
  ret void
}

; GCN-LABEL: {{^}}no_vcc_flat:
; CI: ; NumSgprs: 12
; VI: ; NumSgprs: 14
define void @no_vcc_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7},~{FLAT_SCR}"()
  ret void
}

; GCN-LABEL: {{^}}vcc_flat:
; CI: ; NumSgprs: 12
; VI: ; NumSgprs: 14
define void @vcc_flat() {
entry:
  call void asm sideeffect "", "~{SGPR7},~{VCC},~{FLAT_SCR}"()
  ret void
}
