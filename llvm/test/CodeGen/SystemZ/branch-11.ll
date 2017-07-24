; Test indirect jumps on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

define i32 @f1(i32 %x, i32 %y, i32 %op) {
; CHECK-LABEL: f1:
; CHECK: ahi %r4, -1
; CHECK: clibh %r4, 5, 0(%r14)
; CHECK: llgfr [[OP64:%r[0-5]]], %r4
; CHECK: sllg [[INDEX:%r[1-5]]], [[OP64]], 3
; CHECK: larl [[BASE:%r[1-5]]]
; CHECK: bi 0([[BASE]],[[INDEX]])
entry:
  switch i32 %op, label %exit [
    i32 1, label %b.add
    i32 2, label %b.sub
    i32 3, label %b.and
    i32 4, label %b.or
    i32 5, label %b.xor
    i32 6, label %b.mul
  ]

b.add:
  %add = add i32 %x, %y
  br label %exit

b.sub:
  %sub = sub i32 %x, %y
  br label %exit

b.and:
  %and = and i32 %x, %y
  br label %exit

b.or:
  %or = or i32 %x, %y
  br label %exit

b.xor:
  %xor = xor i32 %x, %y
  br label %exit

b.mul:
  %mul = mul i32 %x, %y
  br label %exit

exit:
  %res = phi i32 [ %x,   %entry ],
                 [ %add, %b.add ],
                 [ %sub, %b.sub ],
                 [ %and, %b.and ],
                 [ %or,  %b.or ],
                 [ %xor, %b.xor ],
                 [ %mul, %b.mul ]
  ret i32 %res
}
