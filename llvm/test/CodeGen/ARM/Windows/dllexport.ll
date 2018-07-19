; RUN: llc -mtriple thumbv7--windows-itanium -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-GNU
; RUN: llc -mtriple thumbv7--windows-gnu -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-GNU
; RUN: llc -mtriple thumbv7--windows-msvc -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-MSVC

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

@r = dllexport alias void (), void () * @f
@s = dllexport alias void (), void () * @g
@t = dllexport alias void (), void () * @f
@u = weak_odr dllexport alias void (), void () * @g

; CHECK: .section .drectve
; CHECK-GNU-NOT: -export:f
; CHECK-GNU: -export:g
; CHECK-GNU-SAME: -export:h
; CHECK-GNU-NOT: -export:i
; CHECK-GNU-SAME: -export:j
; CHECK-GNU-SAME: -export:k
; CHECK-GNU-SAME: -export:l
; CHECK-GNU-SAME: -export:m,data
; CHECK-GNU-SAME: -export:n,data
; CHECK-GNU-SAME: -export:o,data
; CHECK-GNU-SAME: -export:p,data
; CHECK-GNU-SAME: -export:q,data
; CHECK-GNU-SAME: -export:r
; CHECK-GNU-SAME: -export:s
; CHECK-GNU-SAME: -export:t
; CHECK-GNU-SAME: -export:u
; CHECK-MSVC-NOT: /EXPORT:f
; CHECK-MSVC: /EXPORT:g
; CHECK-MSVC-SAME: /EXPORT:h
; CHECK-MSVC-NOT: /EXPORT:i
; CHECK-MSVC-SAME: /EXPORT:j
; CHECK-MSVC-SAME: /EXPORT:k
; CHECK-MSVC-SAME: /EXPORT:l
; CHECK-MSVC-SAME: /EXPORT:m,DATA
; CHECK-MSVC-SAME: /EXPORT:n,DATA
; CHECK-MSVC-SAME: /EXPORT:o,DATA
; CHECK-MSVC-SAME: /EXPORT:p,DATA
; CHECK-MSVC-SAME: /EXPORT:q,DATA
; CHECK-MSVC-SAME: /EXPORT:r
; CHECK-MSVC-SAME: /EXPORT:s
; CHECK-MSVC-SAME: /EXPORT:t
; CHECK-MSVC-SAME: /EXPORT:u

