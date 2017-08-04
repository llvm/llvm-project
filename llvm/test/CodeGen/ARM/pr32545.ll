; RUN: llc %s -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabi"

; CHECK: vld1.16	{[[DREG:d[0-9]+]][0]}, {{.*}}
; CHECK: vmovl.u8	[[QREG:q[0-9]+]], [[DREG]]
; CHECK: vmovl.u16	[[QREG]], [[DREG]]

define void @f(i32 %dstStride, i8* %indvars.iv, <2 x i8>* %zz) {
entry:
  br label %for.body

for.body:
  %tmp = load <2 x i8>, <2 x i8>* %zz, align 1
  %tmp1 = extractelement <2 x i8> %tmp, i32 0
  %.lhs.rhs = zext i8 %tmp1 to i32
  call void @g(i32 %.lhs.rhs)
  br label %for.body
}

declare void @g(i32)
