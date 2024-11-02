;RUN: llc < %s -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs | FileCheck -check-prefixes=CHECK,GCN %s
;RUN: llc < %s -march=r600 -mtriple=r600-- -verify-machineinstrs | FileCheck -check-prefixes=CHECK,R600 %s

%struct.S = type { ptr addrspace(5), ptr addrspace(1), ptr addrspace(4), ptr addrspace(3), ptr, ptr addrspace(2)}

; CHECK-LABEL: nullptr_priv:
; CHECK-NEXT: .long -1
@nullptr_priv = global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5))

; CHECK-LABEL: nullptr_glob:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr_glob = global ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1))

; CHECK-LABEL: nullptr_const:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr_const = global ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4))

; CHECK-LABEL: nullptr_local:
; CHECK-NEXT: .long -1
@nullptr_local = global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3))

; CHECK-LABEL: nullptr_region:
; CHECK-NEXT: .long -1
@nullptr_region = global ptr addrspace(2) addrspacecast (ptr null to ptr addrspace(2))

; CHECK-LABEL: nullptr6:
; R600-NEXT: .long 0
@nullptr6 = global ptr addrspace(6) addrspacecast (ptr null to ptr addrspace(6))

; CHECK-LABEL: nullptr7:
; R600-NEXT: .long 0
@nullptr7 = global ptr addrspace(7) addrspacecast (ptr null to ptr addrspace(7))

; CHECK-LABEL: nullptr8:
; R600-NEXT: .long 0
@nullptr8 = global ptr addrspace(8) addrspacecast (ptr null to ptr addrspace(8))

; CHECK-LABEL: nullptr9:
; R600-NEXT: .long 0
@nullptr9 = global ptr addrspace(9) addrspacecast (ptr null to ptr addrspace(9))

; CHECK-LABEL: nullptr10:
; R600-NEXT: .long 0
@nullptr10 = global ptr addrspace(10) addrspacecast (ptr null to ptr addrspace(10))

; CHECK-LABEL: nullptr11:
; R600-NEXT: .long 0
@nullptr11 = global ptr addrspace(11) addrspacecast (ptr null to ptr addrspace(11))

; CHECK-LABEL: nullptr12:
; R600-NEXT: .long 0
@nullptr12 = global ptr addrspace(12) addrspacecast (ptr null to ptr addrspace(12))

; CHECK-LABEL: nullptr13:
; R600-NEXT: .long 0
@nullptr13 = global ptr addrspace(13) addrspacecast (ptr null to ptr addrspace(13))

; CHECK-LABEL: nullptr14:
; R600-NEXT: .long 0
@nullptr14 = global ptr addrspace(14) addrspacecast (ptr null to ptr addrspace(14))

; CHECK-LABEL: nullptr15:
; R600-NEXT: .long 0
@nullptr15 = global ptr addrspace(15) addrspacecast (ptr null to ptr addrspace(15))

; CHECK-LABEL: nullptr16:
; R600-NEXT: .long 0
@nullptr16 = global ptr addrspace(16) addrspacecast (ptr null to ptr addrspace(16))

; CHECK-LABEL: nullptr17:
; R600-NEXT: .long 0
@nullptr17 = global ptr addrspace(17) addrspacecast (ptr null to ptr addrspace(17))

; CHECK-LABEL: nullptr18:
; R600-NEXT: .long 0
@nullptr18 = global ptr addrspace(18) addrspacecast (ptr null to ptr addrspace(18))

; CHECK-LABEL: nullptr19:
; R600-NEXT: .long 0
@nullptr19 = global ptr addrspace(19) addrspacecast (ptr null to ptr addrspace(19))

; CHECK-LABEL: nullptr20:
; R600-NEXT: .long 0
@nullptr20 = global ptr addrspace(20) addrspacecast (ptr null to ptr addrspace(20))

; CHECK-LABEL: nullptr21:
; R600-NEXT: .long 0
@nullptr21 = global ptr addrspace(21) addrspacecast (ptr null to ptr addrspace(21))

; CHECK-LABEL: nullptr22:
; R600-NEXT: .long 0
@nullptr22 = global ptr addrspace(22) addrspacecast (ptr null to ptr addrspace(22))

; CHECK-LABEL: nullptr23:
; R600-NEXT: .long 0
@nullptr23 = global ptr addrspace(23) addrspacecast (ptr null to ptr addrspace(23))

; CHECK-LABEL: structWithPointers:
; CHECK-NEXT: .long -1
; GCN-NEXT:   .zero 4
; GCN-NEXT:   .quad 0
; R600-NEXT:  .long 0
; GCN-NEXT:   .quad 0
; R600-NEXT:  .long 0
; CHECK-NEXT: .long -1
; GCN-NEXT:   .zero 4
; GCN-NEXT:   .quad 0
; R600-NEXT:  .long 0
; CHECK-NEXT: .long -1
; GCN-NEXT:   .zero 4
@structWithPointers = addrspace(1) global %struct.S {
  ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)),
  ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1)),
  ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4)),
  ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)),
  ptr null,
  ptr addrspace(2) addrspacecast (ptr null to ptr addrspace(2))}, align 4
