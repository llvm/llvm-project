; RUN: llc <%s --mtriple s390x-ibm-zos -emit-gnuas-syntax-on-zos=false | FileCheck %s

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

; CHECK:        ENTRY me1
; CHECK-NEXT: me1 XATTR LINKAGE(XPLINK),REFERENCE(CODE),SCOPE(SECTION)

; CHECK:       ENTRY me2
; CHECK-NEXT: me2 XATTR LINKAGE(XPLINK),REFERENCE(CODE),SCOPE(LIBRARY)

; CHECK:       ENTRY me3
; CHECK-NEXT: me3 XATTR LINKAGE(XPLINK),REFERENCE(CODE),SCOPE(EXPORT)

; CHECK:       EXTRN CELQSTRT
; CHECK-NEXT: CELQSTRT XATTR LINKAGE(OS),REFERENCE(CODE),SCOPE(EXPORT)
; CHECK-NEXT:  WXTRN other1
; CHECK-NEXT: other1 XATTR LINKAGE(XPLINK),REFERENCE(CODE),SCOPE(EXPORT)
; CHECK-NEXT:  EXTRN other2
; CHECK-NEXT: other2 XATTR LINKAGE(XPLINK),REFERENCE(CODE),SCOPE(EXPORT)
; CHECK-NEXT:  END
