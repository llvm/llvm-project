; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 -show-mc-encoding %s -o - \
; RUN:  | FileCheck %s

define i32 @f1(i32 %a.s) {
entry:
; CHECK-LABEL: f1:
; CHECK: it eq
; CHECK: moveq

    %tmp = icmp eq i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f2(i32 %a.s) {
entry:
; CHECK-LABEL: f2:
; CHECK: it gt
; CHECK: movgt
    %tmp = icmp sgt i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f3(i32 %a.s, i32 %b.s) {
entry:
; CHECK-LABEL: f3:
; CHECK: it lt
; CHECK: movlt
    %tmp = icmp slt i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f4(i32 %a.s, i32 %b.s) {
entry:
; CHECK-LABEL: f4:
; CHECK: it le
; CHECK: movle

    %tmp = icmp sle i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f5(i32 %a.u, i32 %b.u) {
entry:
; CHECK-LABEL: f5:
; CHECK: it ls
; CHECK: movls
    %tmp = icmp ule i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f6(i32 %a.u, i32 %b.u) {
entry:
; CHECK-LABEL: f6:
; CHECK: it hi
; CHECK: movhi
    %tmp = icmp ugt i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f7(i32 %a, i32 %b, i32 %c) {
entry:
; CHECK-LABEL: f7:
; CHECK: it hi
; CHECK: lsrhi {{r[0-9]+}}
    %tmp1 = icmp ugt i32 %a, %b
    %tmp2 = udiv i32 %c, 3
    %tmp3 = select i1 %tmp1, i32 %tmp2, i32 3
    ret i32 %tmp3
}

define i32 @f8(i32 %a, i32 %b, i32 %c) {
entry:
; CHECK-LABEL: f8:
; CHECK: it lo
; CHECK: lsllo {{r[0-9]+}}
    %tmp1 = icmp ult i32 %a, %b
    %tmp2 = mul i32 %c, 4
    %tmp3 = select i1 %tmp1, i32 %tmp2, i32 3
    ret i32 %tmp3
}

define i32 @f9(i32 %a, i32 %b, i32 %c) {
entry:
; CHECK-LABEL: f9:
; CHECK: it ge
; CHECK: rorge.w
    %tmp1 = icmp sge i32 %a, %b
    %tmp2 = shl i32 %c, 10
    %tmp3 = lshr i32 %c, 22
    %tmp4 = or i32 %tmp2, %tmp3
    %tmp5 = select i1 %tmp1, i32 %tmp4, i32 3
    ret i32 %tmp5
}

define i32 @f10(i32 %a, i32 %b) {
; CHECK-LABEL: f10:
; CHECK: movwne {{r[0-9]+}}, #1234    @ encoding: [0x40,0xf2,0xd2,0x4{{[0-9a-f]+}}]
    %tst = icmp ne i32 %a, %b
    %val = select i1 %tst, i32 1234, i32 12345
    ret i32 %val
}

; Make sure we pick the Thumb encoding for movw/movt
define i32 @f11(i32 %a, i32 %b) {
; CHECK-LABEL: f11:
; CHECK: movwne {{r[0-9]+}}, #50033         @ encoding: [0x4c,0xf2,0x71,0x3{{[0-9a-f]+}}]
    %tst = icmp ne i32 %a, %b
    %val = select i1 %tst, i32 123454321, i32 543212345
    ret i32 %val
}
