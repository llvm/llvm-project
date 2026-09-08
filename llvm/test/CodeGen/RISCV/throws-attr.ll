; RUN: llc -mtriple=riscv64-unknown-linux-gnu < %s | FileCheck %s

; A function with the throws (herbception) attribute returns its value with a
; discriminant. On RISC-V the discriminant is carried in a2 (X12), leaving the
; value in a0 (and a1 for 128-bit values).

; Success: discriminant is false (0).
define { i64, i1 } @ret_success(i64 %x) #0 {
; CHECK-LABEL: ret_success:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a2, 0
; CHECK-NEXT:    ret
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 false, 1
  ret { i64, i1 } %r
}

; Error: discriminant is true (1).
define { i64, i1 } @ret_error(i64 %x) #0 {
; CHECK-LABEL: ret_error:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a2, 1
; CHECK-NEXT:    ret
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 true, 1
  ret { i64, i1 } %r
}

; 128-bit value uses a0/a1, discriminant in a2.
define { i128, i1 } @ret_i128(i128 %x) #0 {
; CHECK-LABEL: ret_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    li a2, 1
; CHECK-NEXT:    ret
entry:
  %r.i = insertvalue { i128, i1 } poison, i128 %x, 0
  %r = insertvalue { i128, i1 } %r.i, i1 true, 1
  ret { i128, i1 } %r
}

; The caller reads the discriminant from a2 right after the call.
define i64 @call_and_select(i64 %x) #1 {
; CHECK-LABEL: call_and_select:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -16
; CHECK-NEXT:    sd ra, 8(sp)
; CHECK-NEXT:    call ret_error
; CHECK-NEXT:    andi a2, a2, 1
; CHECK-NEXT:    beqz a2, .LBB
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    li a0, 100
; CHECK-NEXT:  .LBB
; CHECK-NEXT:    ld ra, 8(sp)
; CHECK-NEXT:    addi sp, sp, 16
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
