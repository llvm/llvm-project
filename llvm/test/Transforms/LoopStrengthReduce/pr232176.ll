; RUN: opt -S -passes=loop-reduce < %s | FileCheck %s

; SCEV expands each urem in the chain below to x - 1000 * (x /u 1000), so the
; size of the expression doubles with every link. IVUsers should stop at the
; huge expressions that SCEV no longer simplifies: walking them in IVUsers and
; LSR takes exponential time.

target datalayout = "n8:16:32:64"

; CHECK-LABEL: @test(
; CHECK:         %x19 = add i64 %x18, %b19
; CHECK-NEXT:    store i64 %x19, ptr %p
define void @test(i64 %n, ptr %p) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %r0 = urem i64 %i, 1000
  %a0 = add i64 %r0, %i
  %y0 = xor i64 %a0, 1
  %b0 = add i64 %a0, %y0
  %x0 = add i64 %i, %b0
  %r1 = urem i64 %x0, 1000
  %a1 = add i64 %r1, %i
  %y1 = xor i64 %a1, 1
  %b1 = add i64 %a1, %y1
  %x1 = add i64 %x0, %b1
  %r2 = urem i64 %x1, 1000
  %a2 = add i64 %r2, %i
  %y2 = xor i64 %a2, 1
  %b2 = add i64 %a2, %y2
  %x2 = add i64 %x1, %b2
  %r3 = urem i64 %x2, 1000
  %a3 = add i64 %r3, %i
  %y3 = xor i64 %a3, 1
  %b3 = add i64 %a3, %y3
  %x3 = add i64 %x2, %b3
  %r4 = urem i64 %x3, 1000
  %a4 = add i64 %r4, %i
  %y4 = xor i64 %a4, 1
  %b4 = add i64 %a4, %y4
  %x4 = add i64 %x3, %b4
  %r5 = urem i64 %x4, 1000
  %a5 = add i64 %r5, %i
  %y5 = xor i64 %a5, 1
  %b5 = add i64 %a5, %y5
  %x5 = add i64 %x4, %b5
  %r6 = urem i64 %x5, 1000
  %a6 = add i64 %r6, %i
  %y6 = xor i64 %a6, 1
  %b6 = add i64 %a6, %y6
  %x6 = add i64 %x5, %b6
  %r7 = urem i64 %x6, 1000
  %a7 = add i64 %r7, %i
  %y7 = xor i64 %a7, 1
  %b7 = add i64 %a7, %y7
  %x7 = add i64 %x6, %b7
  %r8 = urem i64 %x7, 1000
  %a8 = add i64 %r8, %i
  %y8 = xor i64 %a8, 1
  %b8 = add i64 %a8, %y8
  %x8 = add i64 %x7, %b8
  %r9 = urem i64 %x8, 1000
  %a9 = add i64 %r9, %i
  %y9 = xor i64 %a9, 1
  %b9 = add i64 %a9, %y9
  %x9 = add i64 %x8, %b9
  %r10 = urem i64 %x9, 1000
  %a10 = add i64 %r10, %i
  %y10 = xor i64 %a10, 1
  %b10 = add i64 %a10, %y10
  %x10 = add i64 %x9, %b10
  %r11 = urem i64 %x10, 1000
  %a11 = add i64 %r11, %i
  %y11 = xor i64 %a11, 1
  %b11 = add i64 %a11, %y11
  %x11 = add i64 %x10, %b11
  %r12 = urem i64 %x11, 1000
  %a12 = add i64 %r12, %i
  %y12 = xor i64 %a12, 1
  %b12 = add i64 %a12, %y12
  %x12 = add i64 %x11, %b12
  %r13 = urem i64 %x12, 1000
  %a13 = add i64 %r13, %i
  %y13 = xor i64 %a13, 1
  %b13 = add i64 %a13, %y13
  %x13 = add i64 %x12, %b13
  %r14 = urem i64 %x13, 1000
  %a14 = add i64 %r14, %i
  %y14 = xor i64 %a14, 1
  %b14 = add i64 %a14, %y14
  %x14 = add i64 %x13, %b14
  %r15 = urem i64 %x14, 1000
  %a15 = add i64 %r15, %i
  %y15 = xor i64 %a15, 1
  %b15 = add i64 %a15, %y15
  %x15 = add i64 %x14, %b15
  %r16 = urem i64 %x15, 1000
  %a16 = add i64 %r16, %i
  %y16 = xor i64 %a16, 1
  %b16 = add i64 %a16, %y16
  %x16 = add i64 %x15, %b16
  %r17 = urem i64 %x16, 1000
  %a17 = add i64 %r17, %i
  %y17 = xor i64 %a17, 1
  %b17 = add i64 %a17, %y17
  %x17 = add i64 %x16, %b17
  %r18 = urem i64 %x17, 1000
  %a18 = add i64 %r18, %i
  %y18 = xor i64 %a18, 1
  %b18 = add i64 %a18, %y18
  %x18 = add i64 %x17, %b18
  %r19 = urem i64 %x18, 1000
  %a19 = add i64 %r19, %i
  %y19 = xor i64 %a19, 1
  %b19 = add i64 %a19, %y19
  %x19 = add i64 %x18, %b19
  store i64 %x19, ptr %p
  %i.next = add nuw i64 %i, 1
  %cmp = icmp eq i64 %i.next, %n
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}
