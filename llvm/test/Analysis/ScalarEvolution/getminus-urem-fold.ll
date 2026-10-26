; RUN: opt -passes='print<scalar-evolution>' -disable-output %s 2>&1 | FileCheck %s

; X - C*(X u/ C) folds to (X u% C) for power-of-two C, surfacing range [0, C-1].

; CHECK-LABEL: 'pow2_matches'
; CHECK:        %r = sub i32 %x, %m
; CHECK-NEXT:   -->  (zext i2 (trunc i32 %x to i2) to i32) U: [0,4)
define i32 @pow2_matches(i32 %x) {
  %d = udiv i32 %x, 4
  %m = mul i32 %d, 4
  %r = sub i32 %x, %m
  ret i32 %r
}

; Non-power-of-two C: fold must not match (would recurse via getURemExpr's
; non-pow2 fallback). Existing range analysis still derives [0, C-1].

; CHECK-LABEL: 'non_pow2_blocked'
; CHECK:        %r = sub i32 %x, %m
; CHECK-NEXT:   -->  ((-3 * (%x /u 3)) + %x) U: [0,3)
define i32 @non_pow2_blocked(i32 %x) {
  %d = udiv i32 %x, 3
  %m = mul i32 %d, 3
  %r = sub i32 %x, %m
  ret i32 %r
}
