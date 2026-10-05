; RUN: llc <%s --mtriple s390x-ibm-zos | FileCheck %s

@extint = external global i32, align 4

declare extern_weak void @other1(...)
declare void @other2(...)

define internal signext i32 @me1() {
entry:
  %0 = load i32, ptr @extint, align 4
  ret i32 %0
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
; CHECK-NEXT: C_WSA64 CATTR PART(extint)
; CHECK-NEXT: extint XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
; CHECK-NEXT: extint AMODE 64
; CHECK-NEXT:  WXTRN other1
; CHECK-NEXT: other1 XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
; CHECK-NEXT:  EXTRN other2
; CHECK-NEXT: other2 XATTR LINKAGE(XPLINK),SCOPE(EXPORT)
; CHECK-NEXT:  END
