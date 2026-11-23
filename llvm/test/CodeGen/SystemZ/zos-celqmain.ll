; A module that defines main gets CELQMAIN in RENT format, through which the
; Language Environment startup routine CELQSTRT finds main and its ADA, plus a
; reference to the bootstrap routine CELQBST.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s
; RUN: llc < %s -mtriple=s390x-ibm-zos -filetype=obj -o - | od -Ax -tx1 | FileCheck --check-prefix=OBJ %s

; CHECK:      CELQMAIN XATTR LINKAGE(OS),SCOPE(LIBRARY)
; CHECK-NEXT: CELQMAIN DS 0H
; CHECK-NEXT: * CELQMAIN, RENT format
; CHECK-NEXT:  DC XL4'04000001'
; CHECK-NEXT:  DC XL4'00000000'
; CHECK-NEXT: * Address of main
; CHECK-NEXT:  DC AD(main)
; CHECK-NEXT: * Address of CELQINPL
; CHECK-NEXT:  DC AD(CELQINPL)
; CHECK-NEXT: * Q(environment) of main
; CHECK-NEXT:  DC RD(main)
; CHECK-NEXT: * Reference to CELQBST
; CHECK-NEXT:  DC AD(CELQBST)
; CHECK:       EXTRN CELQINPL
; CHECK:      CELQINPL XATTR LINKAGE(OS),SCOPE(EXPORT)
; CHECK:       EXTRN CELQBST
; CHECK:      CELQBST XATTR LINKAGE(OS),SCOPE(EXPORT)

; OBJ: 04 00 00 01 00 00 00 00

define signext i32 @main() {
  ret i32 42
}
