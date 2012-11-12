; RUN: opt < %s -instcombine -S | FileCheck %s

; rdar://10770603
; (x & y) | (x ^ y) -> x | y 
define i64 @or(i64 %x, i64 %y) nounwind uwtable readnone ssp {
  %1 = and i64 %y, %x
  %2 = xor i64 %y, %x
  %3 = add i64 %1, %2
  ret i64 %3
; CHECK: @or
; CHECK: or i64
; CHECK-NEXT: ret
}

; (x & y) + (x ^ y) -> x | y 
define i64 @or2(i64 %x, i64 %y) nounwind uwtable readnone ssp {
  %1 = and i64 %y, %x
  %2 = xor i64 %y, %x
  %3 = or i64 %1, %2
  ret i64 %3
; CHECK: @or2
; CHECK: or i64
; CHECK-NEXT: ret
}
