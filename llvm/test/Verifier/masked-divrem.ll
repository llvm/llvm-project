; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

; Reject llvm.masked.{u,s}{div,rem} with a non-integer argument.

define <4 x float> @udiv(<4 x float> %x, <4 x float> %y, <4 x i1> %m) {
; CHECK: intrinsic has incorrect argument type!
  %res = call <4 x float> @llvm.masked.udiv(<4 x float> %x, <4 x float> %y, <4 x i1> %m)
  ret <4 x float> %res
}

define <4 x float> @sdiv(<4 x float> %x, <4 x float> %y, <4 x i1> %m) {
; CHECK: intrinsic has incorrect argument type!
  %res = call <4 x float> @llvm.masked.sdiv(<4 x float> %x, <4 x float> %y, <4 x i1> %m)
  ret <4 x float> %res
}

define <4 x float> @urem(<4 x float> %x, <4 x float> %y, <4 x i1> %m) {
; CHECK: intrinsic has incorrect argument type!
  %res = call <4 x float> @llvm.masked.urem(<4 x float> %x, <4 x float> %y, <4 x i1> %m)
  ret <4 x float> %res
}

define <4 x float> @srem(<4 x float> %x, <4 x float> %y, <4 x i1> %m) {
; CHECK: intrinsic has incorrect argument type!
  %res = call <4 x float> @llvm.masked.srem(<4 x float> %x, <4 x float> %y, <4 x i1> %m)
  ret <4 x float> %res
}
