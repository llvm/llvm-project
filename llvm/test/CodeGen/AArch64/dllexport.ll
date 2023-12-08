; RUN: llc -mtriple aarch64-windows-gnu -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-GNU
; RUN: llc -mtriple aarch64-windows-msvc -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-MSVC

define void @f() {
  ret void
}

define dllexport void @g() {
  ret void
}

define dllexport void @h() unnamed_addr {
  ret void
}

declare dllexport void @i()

define linkonce_odr dllexport void @j() {
  ret void
}

define linkonce_odr dllexport void @k() alwaysinline {
  ret void
}

define weak_odr dllexport void @l() {
  ret void
}

@m = dllexport global i32 0, align 4
@n = dllexport unnamed_addr constant i32 0
@o = common dllexport global i32 0, align 4
@p = weak_odr dllexport global i32 0, align 4
@q = weak_odr dllexport unnamed_addr constant i32 0

@r = dllexport alias void (), ptr @f
@s = dllexport alias void (), ptr @g
@t = dllexport alias void (), ptr @f
@u = weak_odr dllexport alias void (), ptr @g

; CHECK: .section .drectve
; CHECK-GNU-NOT: -export:f
; CHECK-GNU: .ascii " -export:g"
; CHECK-GNU: .ascii " -export:h"
; CHECK-GNU-NOT: -export:i
; CHECK-GNU: .ascii " -export:j"
; CHECK-GNU: .ascii " -export:k"
; CHECK-GNU: .ascii " -export:l"
; CHECK-GNU: .ascii " -export:m,data"
; CHECK-GNU: .ascii " -export:n,data"
; CHECK-GNU: .ascii " -export:o,data"
; CHECK-GNU: .ascii " -export:p,data"
; CHECK-GNU: .ascii " -export:q,data"
; CHECK-GNU: .ascii " -export:r"
; CHECK-GNU: .ascii " -export:s"
; CHECK-GNU: .ascii " -export:t"
; CHECK-GNU: .ascii " -export:u"
; CHECK-MSVC-NOT: /EXPORT:f
; CHECK-MSVC: .ascii "  /EXPORT:g"
; CHECK-MSVC: .ascii "  /EXPORT:h"
; CHECK-MSVC-NOT: /EXPORT:i
; CHECK-MSVC: .ascii "  /EXPORT:j"
; CHECK-MSVC: .ascii "  /EXPORT:k"
; CHECK-MSVC: .ascii "  /EXPORT:l"
; CHECK-MSVC: .ascii "  /EXPORT:m,DATA"
; CHECK-MSVC: .ascii "  /EXPORT:n,DATA"
; CHECK-MSVC: .ascii "  /EXPORT:o,DATA"
; CHECK-MSVC: .ascii "  /EXPORT:p,DATA"
; CHECK-MSVC: .ascii "  /EXPORT:q,DATA"
; CHECK-MSVC: .ascii "  /EXPORT:r"
; CHECK-MSVC: .ascii "  /EXPORT:s"
; CHECK-MSVC: .ascii "  /EXPORT:t"
; CHECK-MSVC: .ascii "  /EXPORT:u"
