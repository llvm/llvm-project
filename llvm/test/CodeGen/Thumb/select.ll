; RUN: llc < %s -mtriple=thumb-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=thumb-pc-linux-gnueabi | FileCheck -check-prefix=CHECK-EABI %s

define i32 @f1(i32 %a.s) {
entry:
    %tmp = icmp eq i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK: f1:
; CHECK: beq
; CHECK-EABI: f1:
; CHECK-EABI: beq

define i32 @f2(i32 %a.s) {
entry:
    %tmp = icmp sgt i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK: f2:
; CHECK: bgt
; CHECK-EABI: f2:
; CHECK-EABI: bgt

define i32 @f3(i32 %a.s, i32 %b.s) {
entry:
    %tmp = icmp slt i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK: f3:
; CHECK: blt
; CHECK-EABI: f3:
; CHECK-EABI: blt

define i32 @f4(i32 %a.s, i32 %b.s) {
entry:
    %tmp = icmp sle i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK: f4:
; CHECK: ble
; CHECK-EABI: f4:
; CHECK-EABI: ble

define i32 @f5(i32 %a.u, i32 %b.u) {
entry:
    %tmp = icmp ule i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK: f5:
; CHECK: bls
; CHECK-EABI: f5:
; CHECK-EABI: bls

define i32 @f6(i32 %a.u, i32 %b.u) {
entry:
    %tmp = icmp ugt i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK: f6:
; CHECK: bhi
; CHECK-EABI: f6:
; CHECK-EABI: bhi

define double @f7(double %a, double %b) {
    %tmp = fcmp olt double %a, 1.234e+00
    %tmp1 = select i1 %tmp, double -1.000e+00, double %b
    ret double %tmp1
}
; CHECK: f7:
; CHECK: blt
; CHECK: blt
; CHECK: __ltdf2
; CHECK-EABI: f7:
; CHECK-EABI: __aeabi_dcmplt
; CHECK-EABI: bne
; CHECK-EABI: bne
