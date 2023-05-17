; XFAIL: *
; RUN: llc < %s -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs | FileCheck -check-prefixes=CHECK,GCN %s
; RUN: llc < %s -march=r600 -mtriple=r600-- -verify-machineinstrs | FileCheck -check-prefixes=CHECK,R600 %s

; This is a temporary xfail, as the assembly printer is broken when dealing with
; lowerConstant() trying to return a value of size greater than 8 bytes.

; CHECK-LABEL: nullptr7:
; The exact form of the GCN output depends on how the printer gets fixed.
; GCN-NEXT: .zeroes 5
; R600-NEXT: .long 0
@nullptr7 = global ptr addrspace(7) addrspacecast (ptr null to ptr addrspace(7))

; CHECK-LABEL: nullptr8:
; The exact form of the GCN output depends on how the printer gets fixed.
; GCN-NEXT: .zeroes 4
; R600-NEXT: .long 0
@nullptr8 = global ptr addrspace(8) addrspacecast (ptr null to ptr addrspace(7))
