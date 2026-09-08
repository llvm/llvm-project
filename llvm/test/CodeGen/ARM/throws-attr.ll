; RUN: llc -mtriple=armv7-unknown-linux-gnueabihf < %s | FileCheck %s
; RUN: llc -mtriple=thumbv7-unknown-linux-gnueabihf < %s | FileCheck %s --check-prefix=CHECK-THUMB

; A function with the throws (herbception) attribute returns its value with a
; discriminant. On ARM the discriminant is carried in the CPSR carry flag (C
; bit) instead of a return register.

; Success: discriminant is false (0) -> C is clear.
define { i32, i1 } @ret_success(i32 %x) #0 {
; CHECK-LABEL: ret_success:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    mov r1, #0
; CHECK-NEXT:    subs r1, r1, #1
; CHECK-NEXT:    bx lr
entry:
  %r.i = insertvalue { i32, i1 } poison, i32 %x, 0
  %r = insertvalue { i32, i1 } %r.i, i1 false, 1
  ret { i32, i1 } %r
}

; Error: discriminant is true (1) -> C is set.
define { i32, i1 } @ret_error(i32 %x) #0 {
; CHECK-LABEL: ret_error:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    mov r1, #1
; CHECK-NEXT:    subs r1, r1, #1
; CHECK-NEXT:    bx lr
entry:
  %r.i = insertvalue { i32, i1 } poison, i32 %x, 0
  %r = insertvalue { i32, i1 } %r.i, i1 true, 1
  ret { i32, i1 } %r
}

; The caller reads the discriminant from CPSR.C right after the call (adcs).
define i32 @call_and_select(i32 %x) #1 {
; CHECK-LABEL: call_and_select:
; CHECK:       @ %bb.0:
; CHECK:         bl ret_error
; CHECK-NEXT:    mov r1, #0
; CHECK-NEXT:    adcs r1, r1, #0
; CHECK-THUMB-LABEL: call_and_select:
; CHECK-THUMB:       @ %bb.0:
; CHECK-THUMB:         bl ret_error
; CHECK-THUMB-NEXT:    mov.w r1, #0
; CHECK-THUMB-NEXT:    adcs r1, r1, #0
entry:
  %c = call { i32, i1 } @ret_error(i32 %x)
  %val = extractvalue { i32, i1 } %c, 0
  %disc = extractvalue { i32, i1 } %c, 1
  %sel = select i1 %disc, i32 100, i32 %val
  ret i32 %sel
}

attributes #0 = { throws }
attributes #1 = { nounwind }
