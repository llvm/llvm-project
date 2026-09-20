; Inside an arm of a CSEL, an inner CSEL on the same flags folds to the
; corresponding arm, which breaks the arm's dependency on that inner CSEL.
; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck %s

; The counter-pair idiom: both updates increment the *selected* value, so
; without the fold each CSINC reads the CSEL and serialises on it. After the
; fold each increment reads its own counter and becomes a CINC.
define { i32, i32, i32 } @counter_pair(i8 %p, i32 %a, i32 %b) {
; CHECK-LABEL: counter_pair:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    csel w0, w1, w2, eq
; CHECK-NEXT:    cinc w1, w1, eq
; CHECK-NEXT:    cinc w2, w2, ne
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %sel = select i1 %c, i32 %a, i32 %b
  %inc = add i32 %sel, 1
  %an = select i1 %c, i32 %inc, i32 %a
  %bn = select i1 %c, i32 %b, i32 %inc
  %r0 = insertvalue { i32, i32, i32 } poison, i32 %sel, 0
  %r1 = insertvalue { i32, i32, i32 } %r0, i32 %an, 1
  %r2 = insertvalue { i32, i32, i32 } %r1, i32 %bn, 2
  ret { i32, i32, i32 } %r2
}

; The inverted condition on the inner select selects the other arm.
define i32 @inverted_inner(i8 %p, i32 %a, i32 %b) {
; CHECK-LABEL: inverted_inner:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    cinc w0, w1, eq
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %n = xor i1 %c, true
  %sel = select i1 %n, i32 %b, i32 %a
  %inc = add i32 %sel, 1
  %r = select i1 %c, i32 %inc, i32 %a
  ret i32 %r
}

; A select arm is evaluated unconditionally, so the substitution must not be
; able to introduce a division by zero: here the inner select is what keeps the
; divisor non-zero.
define i32 @no_fold_sdiv(i8 %p, i32 %a, i32 %x) {
; CHECK-LABEL: no_fold_sdiv:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    csinc w8, w1, wzr, eq
; CHECK-NEXT:    sdiv w8, w2, w8
; CHECK-NEXT:    csel w0, w8, w1, eq
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %sel = select i1 %c, i32 %a, i32 1
  %v = sdiv i32 %x, %sel
  %r = select i1 %c, i32 %v, i32 %a
  ret i32 %r
}

; Different flags: the two selects test unrelated conditions, so nothing is
; known about the inner one inside the outer arm.
define i32 @no_fold_other_flags(i8 %p, i8 %q, i32 %a, i32 %b) {
; CHECK-LABEL: no_fold_other_flags:
; CHECK:         tst w1, #0xff
; CHECK-NEXT:    csel w8, w2, w3, eq
; CHECK-NEXT:    tst w0, #0xff
; CHECK-NEXT:    csinc w0, w2, w8, ne
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %d = icmp eq i8 %q, 0
  %sel = select i1 %d, i32 %a, i32 %b
  %inc = add i32 %sel, 1
  %r = select i1 %c, i32 %inc, i32 %a
  ret i32 %r
}

; A multi-use arm that is not an inc/dec shape would have to be duplicated
; without folding into the select, so it is left alone.
define { i32, i32 } @no_fold_multiuse_mul(i8 %p, i32 %a, i32 %b, i32 %x) {
; CHECK-LABEL: no_fold_multiuse_mul:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    csel w8, w1, w2, eq
; CHECK-NEXT:    mul w8, w8, w3
; CHECK-NEXT:    csel w0, w8, w1, eq
; CHECK-NEXT:    csel w1, w2, w8, eq
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %sel = select i1 %c, i32 %a, i32 %b
  %v = mul i32 %sel, %x
  %an = select i1 %c, i32 %v, i32 %a
  %bn = select i1 %c, i32 %b, i32 %v
  %r0 = insertvalue { i32, i32 } poison, i32 %an, 0
  %r1 = insertvalue { i32, i32 } %r0, i32 %bn, 1
  ret { i32, i32 } %r1
}

; A multi-use arm whose constant is not the one CSINC folds stays a real
; instruction in every copy, so duplicating it would be a straight cost.
define { i32, i32 } @no_fold_multiuse_add_big_const(i8 %p, i32 %a, i32 %b) {
; CHECK-LABEL: no_fold_multiuse_add_big_const:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    csel w8, w1, w2, eq
; CHECK-NEXT:    add w8, w8, #1000
; CHECK-NEXT:    csel w0, w8, w1, eq
; CHECK-NEXT:    csel w1, w2, w8, eq
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %sel = select i1 %c, i32 %a, i32 %b
  %v = add i32 %sel, 1000
  %an = select i1 %c, i32 %v, i32 %a
  %bn = select i1 %c, i32 %b, i32 %v
  %r0 = insertvalue { i32, i32 } poison, i32 %an, 0
  %r1 = insertvalue { i32, i32 } %r0, i32 %bn, 1
  ret { i32, i32 } %r1
}

; The other two shapes selection absorbs for free: `xor x, -1` becomes CSINV and
; `sub 0, x` becomes CSNEG, so a copy per use costs nothing and the inner CSEL
; dies.
define { i32, i32 } @multiuse_not(i8 %p, i32 %a, i32 %b) {
; CHECK-LABEL: multiuse_not:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    cinv w0, w1, eq
; CHECK-NEXT:    cinv w1, w2, ne
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %sel = select i1 %c, i32 %a, i32 %b
  %v = xor i32 %sel, -1
  %an = select i1 %c, i32 %v, i32 %a
  %bn = select i1 %c, i32 %b, i32 %v
  %r0 = insertvalue { i32, i32 } poison, i32 %an, 0
  %r1 = insertvalue { i32, i32 } %r0, i32 %bn, 1
  ret { i32, i32 } %r1
}

define { i32, i32 } @multiuse_neg(i8 %p, i32 %a, i32 %b) {
; CHECK-LABEL: multiuse_neg:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    cneg w0, w1, eq
; CHECK-NEXT:    cneg w1, w2, ne
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %sel = select i1 %c, i32 %a, i32 %b
  %v = sub i32 0, %sel
  %an = select i1 %c, i32 %v, i32 %a
  %bn = select i1 %c, i32 %b, i32 %v
  %r0 = insertvalue { i32, i32 } poison, i32 %an, 0
  %r1 = insertvalue { i32, i32 } %r0, i32 %bn, 1
  ret { i32, i32 } %r1
}

; An FP condition needing two condition codes lowers to nested FCSELs on the same
; flags, but the second tests VS rather than EQ or NE, so nothing is known about
; the inner one and the sequence is left alone.
define double @no_fold_fp_two_cc(double %x, double %y, double %a, double %b) {
; CHECK-LABEL: no_fold_fp_two_cc:
; CHECK:         fcmp d0, d1
; CHECK:         fcsel d0, d2, d3, eq
; CHECK-NEXT:    fcsel d0, d2, d0, vs
; CHECK-NEXT:    fadd d0, d0, d4
  %c = fcmp ueq double %x, %y
  %sel = select i1 %c, double %a, double %b
  %v = fadd double %sel, 1.0
  %r = select i1 %c, double %v, double %a
  ret double %r
}

; --- the arm is the inner CSEL directly, with no arithmetic in between -------

; Forwarding the inner arm creates no new nodes, so this needs no cost guard;
; the `CSEL x, x, cc -> x` rule then removes the outer select entirely.
define i32 @direct_inner_arm(i8 %p, i32 %a, i32 %b, i32 %d) {
; CHECK-LABEL: direct_inner_arm:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    csel w0, w1, w3, eq
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %sel = select i1 %c, i32 %a, i32 %b
  %r = select i1 %c, i32 %sel, i32 %d
  ret i32 %r
}

; Same, reached through the inverted condition on the inner select.
define i32 @direct_inner_arm_inverted(i8 %p, i32 %a, i32 %b, i32 %d) {
; CHECK-LABEL: direct_inner_arm_inverted:
; CHECK:         tst w0, #0xff
; CHECK-NEXT:    csel w0, w1, w3, eq
; CHECK-NEXT:    ret
  %c = icmp eq i8 %p, 0
  %n = xor i1 %c, true
  %sel = select i1 %n, i32 %b, i32 %a
  %r = select i1 %c, i32 %sel, i32 %d
  ret i32 %r
}

; CNEG and CSNEG are CSELs over a negation, so a chained absolute value lands in
; the direct-arm case: abs(abs(x)) is x, and the whole sequence disappears.
; Without the fold this stays two CNEGs; folding only the arithmetic arm would
; leave a redundant `csel x0, x8, x0, pl` behind.
define i64 @chained_abs(i64 %x) {
; CHECK-LABEL: chained_abs:
; CHECK-NOT:     cneg
; CHECK:         ret
  %n = sub i64 0, %x
  %c = icmp slt i64 %x, 0
  %a1 = select i1 %c, i64 %n, i64 %x
  %n2 = sub i64 0, %a1
  %a2 = select i1 %c, i64 %n2, i64 %a1
  ret i64 %a2
}

; The inner abs has another use, so only the outer CNEG folds away.
define i64 @chained_abs_live(i64 %x, ptr %p) {
; CHECK-LABEL: chained_abs_live:
; CHECK:         cmp x0, #0
; CHECK-NEXT:    cneg x8, x0, mi
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %n = sub i64 0, %x
  %c = icmp slt i64 %x, 0
  %a1 = select i1 %c, i64 %n, i64 %x
  store i64 %a1, ptr %p
  %n2 = sub i64 0, %a1
  %a2 = select i1 %c, i64 %n2, i64 %a1
  ret i64 %a2
}
