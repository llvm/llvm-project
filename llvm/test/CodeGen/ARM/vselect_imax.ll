; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s
; Make sure that ARM backend with NEON handles vselect.

define void @vmax_v4i32(<4 x i32>* %m, <4 x i32> %a, <4 x i32> %b) {
; CHECK: vcgt.s32 [[QR:q[0-9]+]], [[Q1:q[0-9]+]], [[Q2:q[0-9]+]]
; CHECK: vbsl [[QR]], [[Q1]], [[Q2]]
    %cmpres = icmp sgt <4 x i32> %a, %b
    %maxres = select <4 x i1> %cmpres, <4 x i32> %a,  <4 x i32> %b
    store <4 x i32> %maxres, <4 x i32>* %m
    ret void
}

