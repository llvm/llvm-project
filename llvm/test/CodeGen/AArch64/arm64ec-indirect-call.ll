; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s

define void @simple(ptr %g) {
; CHECK-LABEL:  "#simple":
; CHECK:        str     x30, [sp, #-16]!
; CHECK-NEXT:   .seh_save_reg_x x30, 16
; CHECK-NEXT:   .seh_endprologue
; CHECK-NEXT:   adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:   adrp    x10, $iexit_thunk$cdecl$v$v
; CHECK-NEXT:   add     x10, x10, :lo12:$iexit_thunk$cdecl$v$v
; CHECK-NEXT:   ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:   mov     x11, x0
; CHECK-NEXT:   blr     x8
; CHECK-NEXT:   blr     x11
; CHECK-NEXT:   .seh_startepilogue
; CHECK-NEXT:   ldr     x30, [sp], #16
; CHECK-NEXT:   .seh_save_reg_x x30, 16
; CHECK-NEXT:   .seh_endepilogue
; CHECK-NEXT:   ret

entry:
  call void %g()
  ret void
}

; Make sure the check for the security cookie doesn't use x9.
define void @stackguard(ptr %g) sspreq {
; CHECK-LABEL:  "#stackguard":
; CHECK:          adrp    x8, __os_arm64x_check_icall
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__os_arm64x_check_icall]
; CHECK-NEXT:     blr     x8
; CHECK-NEXT:     adrp    x8, __security_cookie
; CHECK-NEXT:     ldr     x10, [sp, #8]
; CHECK-NEXT:     ldr     x8, [x8, :lo12:__security_cookie]
; CHECK-NEXT:     cmp     x8, x10
; CHECK-NEXT:     b.ne    .LBB1_2
; CHECK-NEXT: // %bb.1:
; CHECK-NEXT:     fmov    d0, #1.00000000
; CHECK-NEXT:     .seh_startepilogue
; CHECK-NEXT:     ldr     x30, [sp, #16]
; CHECK-NEXT:     .seh_save_reg   x30, 16
; CHECK-NEXT:     add     sp, sp, #32
; CHECK-NEXT:     .seh_stackalloc 32
; CHECK-NEXT:     .seh_endepilogue
; CHECK-NEXT:     br      x11

entry:
  %call = tail call double %g(double noundef 1.000000e+00)
  ret void
}
