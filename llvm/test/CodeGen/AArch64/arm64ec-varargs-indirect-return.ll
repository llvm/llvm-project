; RUN: llc < %s -mtriple=arm64ec-pc-windows-msvc -o - | FileCheck %s

;; A variadic function whose value the x64 side returns indirectly still
;; receives three of the four integer arguments in registers, because the
;; hidden return pointer takes the first one.  The fourth argument has to be
;; loaded from the stack, and the callee's view of the stack arguments has to
;; start above it.

target triple = "arm64ec-pc-windows-msvc"

define i128 @ret_i128_vararg(i32 %a, ...) {
  ret i128 0
}

; CHECK:              .def    $ientry_thunk$cdecl$m16$varargs;
; CHECK-LABEL: $ientry_thunk$cdecl$m16$varargs:
; CHECK:       mov     x8, x3
;; The first stack argument fills the last register slot...
; CHECK-NEXT:  ldr     x3, [x4, #32]
; CHECK-NEXT:  mov     x19, x0
;; ...and the remaining stack arguments start after it.
; CHECK-NEXT:  add     x4, x4, #40
; CHECK-NEXT:  mov     x0, x1
; CHECK-NEXT:  mov     x1, x2
; CHECK-NEXT:  mov     x2, x8
; CHECK-NEXT:  mov     x5, xzr
; CHECK-NEXT:  blr     x9
;; The result is written through the hidden return pointer, which is also
;; returned in x8, as the x64 calling convention requires.
; CHECK:       stp     x0, x1, [x19]
; CHECK:       mov     x8, x19

;; A function that returns nothing passes all four integer arguments in
;; registers and only skips over the shadow store.
define void @ret_void_vararg(i32 %a, ...) {
  ret void
}

; CHECK:              .def    $ientry_thunk$cdecl$v$varargs;
; CHECK-LABEL: $ientry_thunk$cdecl$v$varargs:
; CHECK-NOT:   ldr     x3, [x4
; CHECK:       add     x4, x4, #32
; CHECK-NEXT:  mov     x5, xzr
; CHECK-NEXT:  blr     x9
