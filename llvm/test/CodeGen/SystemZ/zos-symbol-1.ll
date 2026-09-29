; RUN: llc <%s --mtriple s390x-ibm-zos | FileCheck %s

declare extern_weak void @other1(...)
declare void @other2(...)

define internal void @me1() {
entry:
  ret void
}

define hidden void @me2() {
entry:
  tail call void @other1()
  ret void
}

define default void @me3() {
entry:
  tail call void @other2()
  ret void
}

; CHECK:      stdin#C CSECT
; CHECK-NEXT: C_CODE64 CATTR ALIGN(3),FILL(0),READONLY,RMODE(64)
; CHECK-NEXT: stdin#C XATTR LINKAGE(XPLINK),PSECT(stdin#S),SCOPE(SECTION)

; CHECK:        ENTRY me1
; CHECK-NEXT: me1 XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(stdin#S),SCOPE(SECTION)

; CHECK:       ENTRY me2
; CHECK-NEXT: me2 XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(stdin#S),SCOPE(LIBRARY)

; CHECK:       ENTRY me3
; CHECK-NEXT: me3 XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(stdin#S),SCOPE(EXPORT)

; CHECK:       EXTRN CELQSTRT
; CHECK-NEXT: CELQSTRT XATTR LINKAGE(OS),SCOPE(EXPORT)
; CHECK-NEXT:  WXTRN other1
; CHECK-NEXT: other1 XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
; CHECK-NEXT:  EXTRN other2
; CHECK-NEXT: other2 XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
; CHECK-NEXT:  END
