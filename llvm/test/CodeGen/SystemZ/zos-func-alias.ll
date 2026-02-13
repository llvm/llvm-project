; Test function aliasing on z/OS
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; CHECK:      ENTRY foo
; CHECK-NEXT: foo XATTR LINKAGE(XPLINK),REFERENCE(CODE),SCOPE(LIBRARY)
; CHECK-NEXT: foo DS 0H
; CHECK-NEXT: ENTRY foo
; CHECK-NEXT: foo1 XATTR LINKAGE(XPLINK),REFERENCE(CODE),SCOPE(LIBRARY)
; CHECK-NEXT: foo1 DS 0H

@foo1 = hidden alias i32 (i32), ptr @foo

define hidden void @foo() {
entry:
  ret void
}
