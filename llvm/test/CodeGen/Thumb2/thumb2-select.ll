; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a.s) {
entry:
; CHECK: f1:
; CHECK: it eq
; CHECK: moveq

    %tmp = icmp eq i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f2(i32 %a.s) {
entry:
; CHECK: f2:
; CHECK: it gt
; CHECK: movgt
    %tmp = icmp sgt i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f3(i32 %a.s, i32 %b.s) {
entry:
; CHECK: f3:
; CHECK: it lt
; CHECK: movlt
    %tmp = icmp slt i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f4(i32 %a.s, i32 %b.s) {
entry:
; CHECK: f4:
; CHECK: it le
; CHECK: movle

    %tmp = icmp sle i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f5(i32 %a.u, i32 %b.u) {
entry:
; CHECK: f5:
; CHECK: it ls
; CHECK: movls
    %tmp = icmp ule i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f6(i32 %a.u, i32 %b.u) {
entry:
; CHECK: f6:
; CHECK: it hi
; CHECK: movhi
    %tmp = icmp ugt i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f7(i32 %a, i32 %b, i32 %c) {
entry:
; CHECK: f7:
; CHECK: it hi
; CHECK: lsrhi.w
    %tmp1 = icmp ugt i32 %a, %b
    %tmp2 = udiv i32 %c, 3
    %tmp3 = select i1 %tmp1, i32 %tmp2, i32 3
    ret i32 %tmp3
}

define i32 @f8(i32 %a, i32 %b, i32 %c) {
entry:
; CHECK: f8:
; CHECK: it lo
; CHECK: lsllo.w
    %tmp1 = icmp ult i32 %a, %b
    %tmp2 = mul i32 %c, 4
    %tmp3 = select i1 %tmp1, i32 %tmp2, i32 3
    ret i32 %tmp3
}

define i32 @f9(i32 %a, i32 %b, i32 %c) {
entry:
; CHECK: f9:
; CHECK: it ge
; CHECK: rorge.w
    %tmp1 = icmp sge i32 %a, %b
    %tmp2 = shl i32 %c, 10
    %tmp3 = lshr i32 %c, 22
    %tmp4 = or i32 %tmp2, %tmp3
    %tmp5 = select i1 %tmp1, i32 %tmp4, i32 3
    ret i32 %tmp5
}
