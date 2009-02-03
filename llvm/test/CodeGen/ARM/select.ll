; RUN: llvm-as < %s | llc -march=arm | grep moveq | count 1
; RUN: llvm-as < %s | llc -march=arm | grep movgt | count 1
; RUN: llvm-as < %s | llc -march=arm | grep movlt | count 3
; RUN: llvm-as < %s | llc -march=arm | grep movle | count 1
; RUN: llvm-as < %s | llc -march=arm | grep movls | count 1
; RUN: llvm-as < %s | llc -march=arm | grep movhi | count 1
; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | \
; RUN:   grep fcpydmi | count 1
; RUN: llvm-as < %s | llc -march=thumb | grep beq | count 1
; RUN: llvm-as < %s | llc -march=thumb | grep bgt | count 1
; RUN: llvm-as < %s | llc -march=thumb | grep blt | count 3
; RUN: llvm-as < %s | llc -march=thumb | grep ble | count 1
; RUN: llvm-as < %s | llc -march=thumb | grep bls | count 1
; RUN: llvm-as < %s | llc -march=thumb | grep bhi | count 1
; RUN: llvm-as < %s | llc -march=thumb | grep __ltdf2

define i32 @f1(i32 %a.s) {
entry:
    %tmp = icmp eq i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f2(i32 %a.s) {
entry:
    %tmp = icmp sgt i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f3(i32 %a.s, i32 %b.s) {
entry:
    %tmp = icmp slt i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f4(i32 %a.s, i32 %b.s) {
entry:
    %tmp = icmp sle i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f5(i32 %a.u, i32 %b.u) {
entry:
    %tmp = icmp ule i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f6(i32 %a.u, i32 %b.u) {
entry:
    %tmp = icmp ugt i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define double @f7(double %a, double %b) {
    %tmp = fcmp olt double %a, 1.234e+00
    %tmp1 = select i1 %tmp, double -1.000e+00, double %b
    ret double %tmp1
}
