; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; A function with the throws (herbception) attribute returns its value with a
; discriminant. On AArch64 the discriminant is carried in the NZCV.C flag
; instead of a return register.

; Success: discriminant is false (0) -> C is clear.
define { i64, i1 } @ret_success(i64 %x) #0 {
; CHECK-LABEL: ret_success:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, wzr
; CHECK-NEXT:    cmp w8, #1
; CHECK-NEXT:    ret
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 false, 1
  ret { i64, i1 } %r
}

; Error: discriminant is true (1) -> C is set.
define { i64, i1 } @ret_error(i64 %x) #0 {
; CHECK-LABEL: ret_error:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #1
; CHECK-NEXT:    cmp w8, #1
; CHECK-NEXT:    ret
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 true, 1
  ret { i64, i1 } %r
}

; The caller reads the discriminant from NZCV.C right after the call (cset hs).
define i64 @call_and_select(i64 %x) #1 {
; CHECK-LABEL: call_and_select:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x30, [sp, #-16]!
; CHECK-NEXT:    bl ret_error
; CHECK-NEXT:    cset w8, hs
; CHECK-NEXT:    tst w8, #0x1
; CHECK-NEXT:    mov w8, #100
; CHECK-NEXT:    csel x0, x8, x0, ne
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    ret
entry:
  %c = call { i64, i1 } @ret_error(i64 %x)
  %val = extractvalue { i64, i1 } %c, 0
  %disc = extractvalue { i64, i1 } %c, 1
  %sel = select i1 %disc, i64 100, i64 %val
  ret i64 %sel
}

attributes #0 = { throws }
attributes #1 = { nounwind }
