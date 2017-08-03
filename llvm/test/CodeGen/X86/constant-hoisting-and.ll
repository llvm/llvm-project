; RUN: llc < %s -O3 -march=x86-64 |FileCheck %s
define i64 @foo(i1 %z, i64 %data1, i64 %data2)
{
; If constant 4294967294 is hoisted to a variable, then we won't be able to use
; the implicit zero extension of 32-bit operations to handle the AND.
entry:
  %val1 = and i64 %data1, 4294967294
  br i1 %z, label %End, label %L_val2

; CHECK: andl    $-2, {{.*}}
; CHECK: andl    $-2, {{.*}}
L_val2:
  %val2 = and i64 %data2, 4294967294
  br label %End

End:
  %p1 = phi i64 [%val1,%entry], [%val2,%L_val2]
  ret i64 %p1
}
