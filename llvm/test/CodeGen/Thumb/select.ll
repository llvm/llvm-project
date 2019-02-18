; RUN: llc < %s -mtriple=thumb-apple-darwin -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=thumb-pc-linux-gnueabi -verify-machineinstrs | FileCheck -check-prefix=CHECK-EABI %s

define i32 @f1(i32 %a.s) {
entry:
    %tmp = icmp eq i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK-LABEL: f1:
; CHECK: beq
; CHECK-EABI-LABEL: f1:
; CHECK-EABI: beq

define i32 @f2(i32 %a.s) {
entry:
    %tmp = icmp sgt i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK-LABEL: f2:
; CHECK: bgt
; CHECK-EABI-LABEL: f2:
; CHECK-EABI: bgt

define i32 @f3(i32 %a.s, i32 %b.s) {
entry:
    %tmp = icmp slt i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK-LABEL: f3:
; CHECK: blt
; CHECK-EABI-LABEL: f3:
; CHECK-EABI: blt

define i32 @f4(i32 %a.s, i32 %b.s) {
entry:
    %tmp = icmp sle i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK-LABEL: f4:
; CHECK: ble
; CHECK-EABI-LABEL: f4:
; CHECK-EABI: ble

define i32 @f5(i32 %a.u, i32 %b.u) {
entry:
    %tmp = icmp ule i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK-LABEL: f5:
; CHECK: bls
; CHECK-EABI-LABEL: f5:
; CHECK-EABI: bls

define i32 @f6(i32 %a.u, i32 %b.u) {
entry:
    %tmp = icmp ugt i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}
; CHECK-LABEL: f6:
; CHECK: bhi
; CHECK-EABI-LABEL: f6:
; CHECK-EABI: bhi

define double @f7(double %a, double %b) {
    %tmp = fcmp olt double %a, 1.234e+00
    %tmp1 = select i1 %tmp, double -1.000e+00, double %b
    ret double %tmp1
}
; CHECK-LABEL: f7:
; CHECK: {{blt|bge}}
; CHECK: {{blt|bge}}
; CHECK: __ltdf2
; CHECK-EABI-LABEL: f7:
; CHECK-EABI: __aeabi_dcmplt
; CHECK-EABI: {{bne|beq}}
; CHECK-EABI: {{bne|beq}}

define {i32, i32} @f8(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
    %cmp = icmp slt i32 %a, %b
    %r1 = select i1 %cmp, i32 %c, i32 %a
    %r2 = select i1 %cmp, i32 %d, i32 %b
    %z = insertvalue { i32, i32 } undef, i32 %r1, 0
    %z2 = insertvalue { i32, i32 } %z, i32 %r2, 1
    ret { i32, i32 } %z2
}

; CHECK-LABEL: f8:
; CHECK: cmp r0, r1
; CHECK: blt
; CHECK: movs
; CHECK: cmp r0, r1
; CHECK: blt
; CHECK: movs
; CHECK: movs
; CHECK: movs
; CHECK: bx lr
