; RUN: opt < %s -instcombine -S | grep select | count 2

; Make sure instcombine don't fold select into operands. We don't want to emit
; select of two integers unless it's selecting 0 / 1.

define i32 @t1(i32 %c, i32 %x) nounwind {
       %t1 = icmp eq i32 %c, 0
       %t2 = lshr i32 %x, 18
       %t3 = select i1 %t1, i32 %t2, i32 %x
       ret i32 %t3
}

define i32 @t2(i32 %c, i32 %x) nounwind {
       %t1 = icmp eq i32 %c, 0
       %t2 = and i32 %x, 18
       %t3 = select i1 %t1, i32 %t2, i32 %x
       ret i32 %t3
}
