; RUN: llc -mtriple s390x-ibm-zos -mcpu=z15 -asm-verbose=false < %s | FileCheck %s

; This tests needs updating when the new external name is emitted into asm output.

; CHECK: L#EPM_sourcename_0 DS 0H
; CHECK-NEXT: DC XL7'00C300C500C500'
; CHECK-NEXT: DC XL1'F1'
; CHECK-NEXT: DC AD(L#PPA1_sourcename_0-L#EPM_sourcename_0)
; CHECK-NEXT: DC XL4'00000008'
; CHECK-NEXT: ENTRY sourcename
; CHECK:sourcename DS 0H
;CHECK-NEXT:  b 2(7)
;CHECK-NEXT: L#sourcename_end_0 DS 0H
;CHECK-NEXT: L#func_end0 DS 0H

define void @sourcename() {
entry:
  ret void
}

; CHECK:var CSECT
; CHECK-NEXT:C_WSA64 CATTR ALIGN(2),FILL(0)
; CHECK:var XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(LIBRARY)
; CHECK-NEXT: DS 0B
; CHECK-NEXT: DC XL4'00000000'

@var = hidden global i32 0, align 4

!zos_mapped_names = !{!2, !3}
!2 = !{!"var", !"other"}
!3 = !{!"sourcename", !"name"}

; CHECK:L#PPA1_sourcename_0 DS 0H
; CHECK-NEXT: DC XL1'02'
; CHECK-NEXT: DC XL1'CE'
; CHECK-NEXT: DC XL2'0000'
; CHECK-NEXT: DC AD(L#PPA2-L#PPA1_sourcename_0)
; CHECK-NEXT: DC XL1'80'
; CHECK-NEXT: DC XL1'80'
; CHECK-NEXT: DC XL1'00'
; CHECK-NEXT: DC XL1'81'
; CHECK-NEXT: DC XL2'0000'
; CHECK-NEXT: DC XL1'00'
; CHECK-NEXT: DC XL1'0'
; CHECK-NEXT: DC AD(L#sourcename_end_0-L#EPM_sourcename_0)
; CHECK-NEXT: DC XL2'0004'
; CHECK-NEXT: DC XL4'95819485'
; CHECK-NEXT: DC AD(L#EPM_sourcename_0-L#PPA1_sourcename_0)
