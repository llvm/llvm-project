; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @t1(i32 %a, i32 %b, i32 %c) nounwind {
; CHECK: t1
; CHECK: mov r0, r1
; CHECK: mvn r1, #-2147483648
; CHECK: cmp r2, #10
; CHECK: it  le
; CHECK: addle r0, r1
        %tmp1 = icmp sgt i32 %c, 10
        %tmp2 = select i1 %tmp1, i32 0, i32 2147483647
        %tmp3 = add i32 %tmp2, %b
        ret i32 %tmp3
}

define i32 @t2(i32 %a, i32 %b, i32 %c) nounwind {
; CHECK: t2
; CHECK: mov r0, r1
; CHECK: cmp r2, #10
; CHECK: it  le
; CHECK: addle.w r0, r0, #-2147483648

        %tmp1 = icmp sgt i32 %c, 10
        %tmp2 = select i1 %tmp1, i32 0, i32 2147483648
        %tmp3 = add i32 %tmp2, %b
        ret i32 %tmp3
}

define i32 @t3(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
; CHECK: t3
; CHECK: mov r0, r1
; CHECK: cmp r2, #10
; CHECK: it  le
; CHECK: suble r0, #10
        %tmp1 = icmp sgt i32 %c, 10
        %tmp2 = select i1 %tmp1, i32 0, i32 10
        %tmp3 = sub i32 %b, %tmp2
        ret i32 %tmp3
}
