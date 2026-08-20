; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

; Functions with different KCFI type identifiers must not be merged.

define internal i32 @a() unnamed_addr !kcfi_type !0 {
; CHECK-LABEL: define internal i32 @a()
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i32 0
; CHECK-NEXT: }
entry:
  ret i32 0
}

define internal i32 @b() unnamed_addr !kcfi_type !1 {
; CHECK-LABEL: define internal i32 @b()
; CHECK-NEXT: entry:
; CHECK-NEXT: ret i32 0
; CHECK-NEXT: }
entry:
  ret i32 0
}

define i32 @caller() {
; CHECK-LABEL: define i32 @caller()
; CHECK-NEXT: entry:
; CHECK-NEXT: %x = call i32 @a()
; CHECK-NEXT: %y = call i32 @b()
; CHECK-NEXT: ret i32 %y
entry:
  %x = call i32 @a()
  %y = call i32 @b()
  ret i32 %y
}

!0 = !{i32 1234}
!1 = !{i32 6789}
