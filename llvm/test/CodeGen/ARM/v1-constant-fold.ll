; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mattr=+v7,+vfp3,-neon | FileCheck %s

; PR15611. Check that we don't crash when constant folding v1i32 types.

; CHECK-LABEL: foo:
define void @foo(i32 %arg) {
bb:
  %tmp = insertelement <4 x i32> undef, i32 %arg, i32 0
  %tmp1 = insertelement <4 x i32> %tmp, i32 0, i32 1
  %tmp2 = insertelement <4 x i32> %tmp1, i32 0, i32 2
  %tmp3 = insertelement <4 x i32> %tmp2, i32 0, i32 3
  %tmp4 = add <4 x i32> %tmp3, <i32 -1, i32 -1, i32 -1, i32 -1>
; CHECK:  bl bar
  call void @bar(<4 x i32> %tmp4)
  ret void
}

declare void @bar(<4 x i32>)
