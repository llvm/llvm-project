; RUN: opt < %s -reassociate -dce -S | FileCheck %s

; MIN_INT cannot be negated during reassociation

define i32 @minint(i32 %i) {
; CHECK:  %mul = mul i32 %i, -2147483648
; CHECK-NEXT:  %add = add i32 %mul, 1
; CHECK-NEXT:  ret i32 %add
  %mul = mul i32 %i, -2147483648
  %add = add i32 %mul, 1
  ret i32 %add
}

