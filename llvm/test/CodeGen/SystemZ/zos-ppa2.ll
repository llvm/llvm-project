; RUN: sed -e 's/!"MODE"/!"ascii"/' -e 's/BYTE/85/' %s > %t.ascii.ll
; RUN: llc -mtriple s390x-ibm-zos -mcpu=z15 -asm-verbose=true < %t.ascii.ll | \
; RUN: FileCheck %t.ascii.ll

; RUN: sed -e 's/!"MODE"/!"ebcdic"/' -e 's/BYTE/81/' %s > %t.ebcdic.ll
; RUN: llc -mtriple s390x-ibm-zos -mcpu=z15 -asm-verbose=true < %t.ebcdic.ll | \
; RUN: FileCheck %t.ebcdic.ll

; CHECK: C_CODE64 CATTR
; CHECK: L#PPA2 DS 0H
; CHECK:  DC XL1'03'
; CHECK:  DC XL1'E7'
; CHECK:  DC XL1'22'
; CHECK:  DC XL1'04'
; CHECK:  DC AD(CELQSTRT-L#PPA2)
; CHECK:  DC XL4'00000000'
; CHECK:  DC AD(L#DVS-L#PPA2)
; CHECK:  DC XL4'00000000'
; CHECK:  DC XL1'BYTE'
; CHECK:  DC XL1'00'
; CHECK:  DC XL2'0000'
; CHECK: L#DVS DS 0H
; CHECK:  DC XL14'F1F9F7F0F0F1F0F1F0F0F0F0F0F0'
; CHECK:  DC XL6'F2F3F0F0F0F0'
; CHECK:  DC XL2'0000'

; CHECK: C_@@QPPA2 CATTR ALIGN(3),FILL(0),NOTEXECUTABLE,READONLY,RMODE(64),PART(.
; CHECK:                .&ppa2)
; CHECK: .&ppa2 XATTR LINKAGE(OS),REFERENCE(DATA),SCOPE(SECTION)
; CHECK: * A(PPA2-CELQSTRT)
; CHECK:  DC AD(L#PPA2-CELQSTRT)

; CHECK: L#EPM_void_test_0 DS 0H
; CHECK: * Offset to PPA1
; CHECK:  DC AD(L#PPA1_void_test_0-L#EPM_void_test_0)
 
; CHECK: B_IDRL CATTR ALIGN(3),FILL(0),NOLOAD,READONLY,RMODE(64)
; CHECK:  DC XL1'00'
; CHECK:  DC XL1'03'
; CHECK:  DC XL2'001E'
; CHECK:  DC XL30'D3D3E5D4404040404040{{([[:xdigit:]]{8})}}F1F9F7F0F0F1F0F1F0F0F0F0F0F0F0F0'

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"zos_le_char_mode", !"MODE"}

define void @void_test() {
entry:
  ret void
}
