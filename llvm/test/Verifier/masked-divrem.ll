; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

; Reject llvm.masked.{u,s}{div,rem} with a non-integer argument.

declare <4 x float> @llvm.masked.udiv.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x float> @llvm.masked.sdiv.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x float> @llvm.masked.urem.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x float> @llvm.masked.srem.v4f32(<4 x float>, <4 x float>, <4 x i1>)

define <4 x float> @udiv(<4 x float> %x, <4 x float> %y, <4 x i1> %m) {
; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
  %res = call <4 x float> @llvm.masked.udiv.v4f32(<4 x float> %x, <4 x float> %y, <4 x i1> %m)
  ret <4 x float> %res
}

define <4 x float> @sdiv(<4 x float> %x, <4 x float> %y, <4 x i1> %m) {
; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
  %res = call <4 x float> @llvm.masked.sdiv.v4f32(<4 x float> %x, <4 x float> %y, <4 x i1> %m)
  ret <4 x float> %res
}

define <4 x float> @urem(<4 x float> %x, <4 x float> %y, <4 x i1> %m) {
; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
  %res = call <4 x float> @llvm.masked.urem.v4f32(<4 x float> %x, <4 x float> %y, <4 x i1> %m)
  ret <4 x float> %res
}

define <4 x float> @srem(<4 x float> %x, <4 x float> %y, <4 x i1> %m) {
; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
  %res = call <4 x float> @llvm.masked.srem.v4f32(<4 x float> %x, <4 x float> %y, <4 x i1> %m)
  ret <4 x float> %res
}
