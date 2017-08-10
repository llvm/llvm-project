; RUN: llc < %s -O3 -mtriple=x86_64-- |FileCheck %s
define i64 @foo(i64 %data1, i64 %data2, i64 %data3)
{
; If constant 4294967295 is hoisted to a variable, then we won't be able to
; use a shift right by 32 to optimize the compare.
entry:
  %val1 = add i64 %data3, 1
  %x = icmp ugt i64 %data1, 4294967295
  br i1 %x, label %End, label %L_val2

; CHECK: shrq    $32, {{.*}}
; CHECK: shrq    $32, {{.*}}
L_val2:
  %val2 = add i64 %data3, 2
  %y = icmp ugt i64 %data2, 4294967295
  br i1 %y, label %End, label %L_val3

L_val3:
  %val3 = add i64 %data3, 3
  br label %End

End:
  %p1 = phi i64 [%val1,%entry], [%val2,%L_val2], [%val3,%L_val3]
  ret i64 %p1
}
