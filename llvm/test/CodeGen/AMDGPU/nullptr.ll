;RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck %s

%struct.S = type { i32*, i32 addrspace(1)*, i32 addrspace(2)*, i32 addrspace(3)*, i32 addrspace(4)*, i32 addrspace(5)*}

; CHECK-LABEL: nullptr_priv:
; CHECK-NEXT: .long -1
@nullptr_priv = global i32* addrspacecast (i32 addrspace(4)* null to i32*)

; CHECK-LABEL: nullptr_glob:
; CHECK-NEXT: .quad 0
@nullptr_glob = global i32 addrspace(1)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(1)*)

; CHECK-LABEL: nullptr_const:
; CHECK-NEXT: .quad 0
@nullptr_const = global i32 addrspace(2)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(2)*)

; CHECK-LABEL: nullptr_local:
; CHECK-NEXT: .long -1
@nullptr_local = global i32 addrspace(3)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(3)*)

; CHECK-LABEL: nullptr_region:
; CHECK-NEXT: .long -1
@nullptr_region = global i32 addrspace(5)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(5)*)

; CHECK-LABEL: nullptr6:
; CHECK-NEXT: .long 0
@nullptr6 = global i32 addrspace(6)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(6)*)

; CHECK-LABEL: nullptr7:
; CHECK-NEXT: .long 0
@nullptr7 = global i32 addrspace(7)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(7)*)

; CHECK-LABEL: nullptr8:
; CHECK-NEXT: .long 0
@nullptr8 = global i32 addrspace(8)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(8)*)

; CHECK-LABEL: nullptr9:
; CHECK-NEXT: .long 0
@nullptr9 = global i32 addrspace(9)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(9)*)

; CHECK-LABEL: nullptr10:
; CHECK-NEXT: .long 0
@nullptr10 = global i32 addrspace(10)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(10)*)

; CHECK-LABEL: nullptr11:
; CHECK-NEXT: .long 0
@nullptr11 = global i32 addrspace(11)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(11)*)

; CHECK-LABEL: nullptr12:
; CHECK-NEXT: .long 0
@nullptr12 = global i32 addrspace(12)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(12)*)

; CHECK-LABEL: nullptr13:
; CHECK-NEXT: .long 0
@nullptr13 = global i32 addrspace(13)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(13)*)

; CHECK-LABEL: nullptr14:
; CHECK-NEXT: .long 0
@nullptr14 = global i32 addrspace(14)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(14)*)

; CHECK-LABEL: nullptr15:
; CHECK-NEXT: .long 0
@nullptr15 = global i32 addrspace(15)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(15)*)

; CHECK-LABEL: nullptr16:
; CHECK-NEXT: .long 0
@nullptr16 = global i32 addrspace(16)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(16)*)

; CHECK-LABEL: nullptr17:
; CHECK-NEXT: .long 0
@nullptr17 = global i32 addrspace(17)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(17)*)

; CHECK-LABEL: nullptr18:
; CHECK-NEXT: .long 0
@nullptr18 = global i32 addrspace(18)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(18)*)

; CHECK-LABEL: nullptr19:
; CHECK-NEXT: .long 0
@nullptr19 = global i32 addrspace(19)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(19)*)

; CHECK-LABEL: nullptr20:
; CHECK-NEXT: .long 0
@nullptr20 = global i32 addrspace(20)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(20)*)

; CHECK-LABEL: nullptr21:
; CHECK-NEXT: .long 0
@nullptr21 = global i32 addrspace(21)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(21)*)

; CHECK-LABEL: nullptr22:
; CHECK-NEXT: .long 0
@nullptr22 = global i32 addrspace(22)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(22)*)

; CHECK-LABEL: nullptr23:
; CHECK-NEXT: .long 0
@nullptr23 = global i32 addrspace(23)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(23)*)

; CHECK-LABEL: structWithPointers:
; CHECK-NEXT: .long -1
; CHECK-NEXT: .zero 4
; CHECK-NEXT: .quad 0
; CHECK-NEXT: .quad 0
; CHECK-NEXT: .long -1
; CHECK-NEXT: .zero 4
; CHECK-NEXT: .quad 0
; CHECK-NEXT: .long -1
; CHECK-NEXT: .zero 4
@structWithPointers = addrspace(1) global %struct.S {
  i32* addrspacecast (i32 addrspace(4)* null to i32*),
  i32 addrspace(1)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(1)*),
  i32 addrspace(2)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(2)*),
  i32 addrspace(3)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(3)*),
  i32 addrspace(4)* null,
  i32 addrspace(5)* addrspacecast (i32 addrspace(4)* null to i32 addrspace(5)*)}, align 4
