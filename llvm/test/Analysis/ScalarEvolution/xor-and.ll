; RUN: opt < %s -scalar-evolution -analyze \
; RUN:   | grep "\-->  (zext i4 (-8 + (trunc i64 (8 \* %x) to i4)) to i64)"

; ScalarEvolution shouldn't try to analyze %z into something like
;   -->  (zext i4 (-1 + (-1 * (trunc i64 (8 * %x) to i4))) to i64)

define i64 @foo(i64 %x) {
  %a = shl i64 %x, 3
  %t = and i64 %a, 8
  %z = xor i64 %t, 8
  ret i64 %z
}
