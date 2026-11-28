;RUN: llc < %s -mtriple=amdgpu6.00-- | FileCheck -check-prefixes=CHECK,GCN %s

%struct.S = type { ptr addrspace(5), ptr addrspace(1), ptr addrspace(4), ptr addrspace(3), ptr, ptr addrspace(2)}

; A plain null pointer has the null value of the data layout, which is zero in
; all address spaces. A cast of the flat null pointer has the null value of the
; target, which is -1 in some address spaces.

; CHECK-LABEL: nullptr_flat_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr_flat_null = global ptr null

; CHECK-LABEL: nullptr_priv:
; CHECK-NEXT: .long -1
@nullptr_priv = global ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5))

; CHECK-LABEL: nullptr_priv_null:
; CHECK-NEXT: .long 0
@nullptr_priv_null = global ptr addrspace(5) null

; CHECK-LABEL: nullptr_glob:
; GCN-NEXT: .quad 0
@nullptr_glob = global ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1))

; CHECK-LABEL: nullptr_glob_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr_glob_null = global ptr addrspace(1) null

; CHECK-LABEL: nullptr_const:
; GCN-NEXT: .quad 0
@nullptr_const = global ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4))

; CHECK-LABEL: nullptr_const_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr_const_null = global ptr addrspace(4) null

; CHECK-LABEL: nullptr_local:
; CHECK-NEXT: .long -1
@nullptr_local = global ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3))

; CHECK-LABEL: nullptr_local_null:
; CHECK-NEXT: .long 0
@nullptr_local_null = global ptr addrspace(3) null

; CHECK-LABEL: nullptr_region:
; CHECK-NEXT: .long -1
@nullptr_region = global ptr addrspace(2) addrspacecast (ptr null to ptr addrspace(2))

; CHECK-LABEL: nullptr_region_null:
; CHECK-NEXT: .long 0
@nullptr_region_null = global ptr addrspace(2) null

; CHECK-LABEL: nullptr6:
@nullptr6 = global ptr addrspace(6) addrspacecast (ptr null to ptr addrspace(6))

; CHECK-LABEL: nullptr6_null:
; CHECK-NEXT: .long 0
@nullptr6_null = global ptr addrspace(6) null

; FIXME: AsmPrinter can't handle 128-bit constants
; FIXME-LABEL: nullptr7:
; FIXME-NEXT: .long 0
; FIXME @nullptr7 = global ptr addrspace(7) addrspacecast (ptr null to ptr addrspace(7))

; FIXME-LABEL: nullptr8:
; FIXME-NEXT: .long 0
; FIXME @nullptr8 = global ptr addrspace(8) addrspacecast (ptr null to ptr addrspace(8))

; FIXME-LABEL: nullptr9:
; FIXME-NEXT: .long 0
; FIXME @nullptr9 = global ptr addrspace(9) addrspacecast (ptr null to ptr addrspace(9))

; A null constant wider than 64 bits.
; CHECK-LABEL: nullptr8_null:
; GCN-NEXT: .zero 16
; R600-NEXT: .long 0
@nullptr8_null = global ptr addrspace(8) null

; A null constant with tail padding.
; CHECK-LABEL: nullptr9_null:
; GCN-NEXT: .zero 24
; GCN-NEXT: .zero 8
; R600-NEXT: .long 0
@nullptr9_null = global ptr addrspace(9) null

; CHECK-LABEL: nullptr10:
@nullptr10 = global ptr addrspace(10) addrspacecast (ptr null to ptr addrspace(10))

; CHECK-LABEL: nullptr10_null:
; CHECK-NEXT: .long 0
@nullptr10_null = global ptr addrspace(10) null

; CHECK-LABEL: nullptr11:
@nullptr11 = global ptr addrspace(11) addrspacecast (ptr null to ptr addrspace(11))

; CHECK-LABEL: nullptr11_null:
; CHECK-NEXT: .long 0
@nullptr11_null = global ptr addrspace(11) null

; CHECK-LABEL: nullptr12:
@nullptr12 = global ptr addrspace(12) addrspacecast (ptr null to ptr addrspace(12))

; CHECK-LABEL: nullptr12_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr12_null = global ptr addrspace(12) null

; Address space 13 is the VGPR address space, whose null pointer is -1 like the
; other address spaces where zero is a valid address.
; CHECK-LABEL: nullptr13:
; CHECK-NEXT: .long -1
@nullptr13 = global ptr addrspace(13) addrspacecast (ptr null to ptr addrspace(13))

; CHECK-LABEL: nullptr13_null:
; CHECK-NEXT: .long 0
@nullptr13_null = global ptr addrspace(13) null

; CHECK-LABEL: nullptr14:
@nullptr14 = global ptr addrspace(14) addrspacecast (ptr null to ptr addrspace(14))

; CHECK-LABEL: nullptr14_null:
; CHECK-NEXT: .long 0
@nullptr14_null = global ptr addrspace(14) null

; CHECK-LABEL: nullptr15:
@nullptr15 = global ptr addrspace(15) addrspacecast (ptr null to ptr addrspace(15))

; CHECK-LABEL: nullptr15_null:
; CHECK-NEXT: .long 0
@nullptr15_null = global ptr addrspace(15) null

; CHECK-LABEL: nullptr16:
@nullptr16 = global ptr addrspace(16) addrspacecast (ptr null to ptr addrspace(16))

; CHECK-LABEL: nullptr16_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr16_null = global ptr addrspace(16) null

; CHECK-LABEL: nullptr17:
@nullptr17 = global ptr addrspace(17) addrspacecast (ptr null to ptr addrspace(17))

; CHECK-LABEL: nullptr17_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr17_null = global ptr addrspace(17) null

; CHECK-LABEL: nullptr18:
@nullptr18 = global ptr addrspace(18) addrspacecast (ptr null to ptr addrspace(18))

; CHECK-LABEL: nullptr18_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr18_null = global ptr addrspace(18) null

; CHECK-LABEL: nullptr19:
@nullptr19 = global ptr addrspace(19) addrspacecast (ptr null to ptr addrspace(19))

; CHECK-LABEL: nullptr19_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr19_null = global ptr addrspace(19) null

; CHECK-LABEL: nullptr20:
@nullptr20 = global ptr addrspace(20) addrspacecast (ptr null to ptr addrspace(20))

; CHECK-LABEL: nullptr20_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr20_null = global ptr addrspace(20) null

; CHECK-LABEL: nullptr21:
@nullptr21 = global ptr addrspace(21) addrspacecast (ptr null to ptr addrspace(21))

; CHECK-LABEL: nullptr21_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr21_null = global ptr addrspace(21) null

; CHECK-LABEL: nullptr22:
@nullptr22 = global ptr addrspace(22) addrspacecast (ptr null to ptr addrspace(22))

; CHECK-LABEL: nullptr22_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr22_null = global ptr addrspace(22) null

; CHECK-LABEL: nullptr23:
@nullptr23 = global ptr addrspace(23) addrspacecast (ptr null to ptr addrspace(23))

; CHECK-LABEL: nullptr23_null:
; GCN-NEXT: .quad 0
; R600-NEXT: .long 0
@nullptr23_null = global ptr addrspace(23) null

; CHECK-LABEL: structWithPointers:
; CHECK-NEXT: .long -1
; GCN-NEXT:   .zero 4
; GCN-NEXT:   .quad 0
; GCN-NEXT:   .quad 0
; CHECK-NEXT: .long -1
; GCN-NEXT:   .zero 4
; GCN-NEXT:   .quad 0
; CHECK-NEXT: .long -1
; GCN-NEXT:   .zero 4
@structWithPointers = addrspace(1) global %struct.S {
  ptr addrspace(5) addrspacecast (ptr null to ptr addrspace(5)),
  ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1)),
  ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4)),
  ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)),
  ptr null,
  ptr addrspace(2) addrspacecast (ptr null to ptr addrspace(2))}, align 4

; CHECK-LABEL: structWithWidePointer:
; CHECK-NEXT: .long 1
; GCN-NEXT:   .zero 12
; GCN-NEXT:   .zero 16
; R600-NEXT:  .long 0
@structWithWidePointer = addrspace(1) global { i32, ptr addrspace(8) } { i32 1, ptr addrspace(8) null }
