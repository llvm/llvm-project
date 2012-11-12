; RUN: llc < %s -march=arm | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = and i32 %a, %tmp
    ret i32 %tmp1
}

; CHECK: bic	r0, r0, r1

define i32 @f2(i32 %a, i32 %b) {
    %tmp = xor i32 %b, 4294967295
    %tmp1 = and i32 %tmp, %a
    ret i32 %tmp1
}

; CHECK: bic	r0, r0, r1
