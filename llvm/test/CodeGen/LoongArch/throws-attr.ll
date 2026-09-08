; RUN: llc -mtriple=loongarch64-unknown-linux-gnu < %s | FileCheck %s

; A function with the throws (herbception) attribute returns its value with a
; discriminant. On LoongArch the discriminant is carried in a2 (R6), leaving
; the value in a0 (and a1 for 128-bit values).

; Success: discriminant is false (0).
define { i64, i1 } @ret_success(i64 %x) #0 {
; CHECK-LABEL: ret_success:
; CHECK:       # %bb.0:
; CHECK-NEXT:    move $a2, $zero
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
; CHECK-NEXT:    ori $a2, $zero, 1
; CHECK-NEXT:    ret
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 true, 1
  ret { i64, i1 } %r
}

; The caller reads the discriminant from a2 right after the call.
define i64 @call_and_select(i64 %x) #1 {
; CHECK-LABEL: call_and_select:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi.d $sp, $sp, -16
; CHECK-NEXT:    st.d $ra, $sp, 8
; CHECK-NEXT:    pcaddu18i $ra, %call36(ret_error)
; CHECK-NEXT:    jirl $ra, $ra, 0
; CHECK-NEXT:    andi $a1, $a2, 1
; CHECK-NEXT:    masknez $a0, $a0, $a1
; CHECK-NEXT:    ori $a2, $zero, 100
; CHECK-NEXT:    maskeqz $a1, $a2, $a1
; CHECK-NEXT:    or $a0, $a1, $a0
; CHECK-NEXT:    ld.d $ra, $sp, 8
; CHECK-NEXT:    addi.d $sp, $sp, 16
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
