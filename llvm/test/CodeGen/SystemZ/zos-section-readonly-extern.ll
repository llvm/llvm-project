; Externally visible read-only data is placed into the WSA (C_WSA64), because
; a reference from another translation unit is always a part in the WSA.
; Local read-only data stays in the code section.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; CHECK:      tab CSECT
; CHECK-NEXT: C_WSA64 CATTR {{.*}}PART(t
; CHECK:      tab XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(LIBRARY)
; CHECK:      exp CSECT
; CHECK-NEXT: C_WSA64 CATTR {{.*}}PART(e
; CHECK:      exp XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(EXPORT)
; CHECK:      C_CODE64 CATTR
; CHECK:      loc XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
; CHECK-NEXT: loc DS 0H

@tab = hidden constant [2 x i32] [i32 1, i32 2], align 4
@exp = constant [2 x i32] [i32 3, i32 4], align 4
@loc = internal constant [2 x i32] [i32 5, i32 6], align 4

define signext i32 @get(i64 %i) {
  %p = getelementptr inbounds [2 x i32], ptr @loc, i64 0, i64 %i
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
