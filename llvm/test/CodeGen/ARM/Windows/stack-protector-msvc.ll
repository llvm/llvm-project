; RUN: llc -mtriple=thumbv7-windows-msvc < %s -o - | FileCheck --check-prefix=MSVC %s
; RUN: llc -mtriple=thumbv7-windows-msvc -O0 < %s -o - | FileCheck --check-prefix=MSVC %s

define void @_Z1fv() sspreq {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @_Z7CapturePi(i32* nonnull %x)
  ret void
}

declare void @_Z7CapturePi(i32*)

; MSVC: movw r0, :lower16:__security_cookie
; MSVC: movt r0, :upper16:__security_cookie
; MSVC: ldr r0, [r0]
; MSVC: str r0, [sp, #4]
; MSVC: bl  _Z7CapturePi
; MSVC: ldr r0, [sp, #4]
; MSVC: bl  __security_check_cookie
