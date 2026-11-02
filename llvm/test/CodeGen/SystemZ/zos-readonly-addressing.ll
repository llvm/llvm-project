; Read-only data stays in the code section (and is addressed PC-relative) only
; if it is local and its initializer needs no relocations. Externally visible
; read-only data and tables of function pointers (which point to function
; descriptors in the WSA) are parts in the WSA and are addressed via the ADA.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; CHECK-LABEL: useext DS 0H
; CHECK:         lg 2,0(5)
; CHECK-NOT:     larl
; CHECK:         b 2(7)
; CHECK-LABEL: useloc DS 0H
; CHECK:         larl 2,loc
; CHECK:         b 2(7)
; CHECK-LABEL: usetbl DS 0H
; CHECK:         lg 1,8(5)
; CHECK-NOT:     larl
; CHECK:         b 2(7)
; CHECK:      ext CSECT
; CHECK-NEXT: C_WSA64 CATTR {{.*}}PART(ext)
; CHECK:      loc XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(SECTION)
; CHECK-NEXT: loc DS 0H
; CHECK:      tbl CSECT
; CHECK-NEXT: C_WSA64 CATTR {{.*}}PART(tbl)
@ext = constant [2 x i32] [i32 3, i32 4], align 4
@loc = internal constant [2 x i32] [i32 5, i32 6], align 4
@tbl = internal constant [1 x ptr] [ptr @f], align 8
define internal void @f() { ret void }
define signext i32 @useext(i64 %i) {
  %p = getelementptr inbounds [2 x i32], ptr @ext, i64 0, i64 %i
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
define signext i32 @useloc(i64 %i) {
  %p = getelementptr inbounds [2 x i32], ptr @loc, i64 0, i64 %i
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
define ptr @usetbl() {
  %v = load ptr, ptr @tbl, align 8
  ret ptr %v
}
