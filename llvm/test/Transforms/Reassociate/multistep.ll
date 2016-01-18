; RUN: opt < %s -reassociate -S | FileCheck %s

define i64 @multistep1(i64 %a, i64 %b, i64 %c) {
; Check that a*a*b+a*a*c is turned into a*(a*(b+c)).
; CHECK-LABEL: @multistep1(
  %t0 = mul i64 %a, %b
  %t1 = mul i64 %a, %t0 ; a*(a*b)
  %t2 = mul i64 %a, %c
  %t3 = mul i64 %a, %t2 ; a*(a*c)
  %t4 = add i64 %t1, %t3
; CHECK-NEXT: add i64 %c, %b
; CHECK-NEXT: mul i64 %a, %tmp{{.*}}
; CHECK-NEXT: mul i64 %tmp{{.*}}, %a
; CHECK-NEXT: ret
  ret i64 %t4
}

define i64 @multistep2(i64 %a, i64 %b, i64 %c, i64 %d) {
; Check that a*b+a*c+d is turned into a*(b+c)+d.
; CHECK-LABEL: @multistep2(
  %t0 = mul i64 %a, %b
  %t1 = mul i64 %a, %c
  %t2 = add i64 %t1, %d ; a*c+d
  %t3 = add i64 %t0, %t2 ; a*b+(a*c+d)
; CHECK-NEXT: add i64 %c, %b
; CHECK-NEXT: mul i64 %tmp{{.*}}, %a
; CHECK-NEXT: add i64 %tmp{{.*}}, %d
; CHECK-NEXT: ret
  ret i64 %t3
}
