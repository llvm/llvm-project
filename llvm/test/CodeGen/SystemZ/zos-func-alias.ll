; Test function aliasing on z/OS
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; CHECK:      ENTRY foo
; CHECK-NEXT: foo XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(stdin#S),SCOPE(LIBRARY)
; CHECK-NEXT: foo DS 0H
; CHECK-NEXT: ENTRY bar
; CHECK-NEXT: bar XATTR LINKAGE(XPLINK),REFERENCE(CODE),PSECT(stdin#S),SCOPE(LIBRARY)
; CHECK-NEXT: bar DS 0H

@bar = hidden alias i32 (i32), ptr @foo

define hidden void @foo() {
entry:
  ret void
}
