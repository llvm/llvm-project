; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

define i32 @sadd_arg_int(float %x, i32 %y) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call i32 @llvm.sadd.sat.i32(float %x, i32 %y)
  ret i32 %r
}

define i37 @uadd_arg_int(half %x, i37 %y) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call i37 @llvm.uadd.sat.i37(i37 %y, half %x)
  ret i37 %r
}

define <4 x i32> @ssub_arg_int(<5 x i32> %x, <4 x i32> %y) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call <4 x i32> @llvm.ssub.sat.v4i32(<5 x i32> %x, <4 x i32> %y)
  ret <4 x i32> %r
}

define <3 x i37> @usub_arg_int(<3 x i37> %x, <3 x i32> %y) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call <3 x i37> @llvm.usub.sat.v3i37(<3 x i37> %x, <3 x i32> %y)
  ret <3 x i37> %r
}

define float @ushl_return_int(i32 %x, i32 %y) {
; CHECK: Intrinsic has incorrect return type!
  %r = call float @llvm.ushl.sat.i32(i32 %x, i32 %y)
  ret float %r
}

define <4 x float> @sshl_return_int_vec(<4 x i32> %x, <4 x i32> %y) {
; CHECK: Intrinsic has incorrect return type!
  %r = call <4 x float> @llvm.sshl.sat.v4i32(<4 x i32> %x, <4 x i32> %y)
  ret <4 x float> %r
}

declare i32 @llvm.sadd.sat.i32(float, i32)
declare i37 @llvm.uadd.sat.i37(i37, half)
declare <4 x i32> @llvm.ssub.sat.v4i32(<5 x i32>, <4 x i32>)
declare <3 x i37> @llvm.usub.sat.v3i37(<3 x i37>, <3 x i32>)
declare float @llvm.ushl.sat.i32(i32, i32)
declare <4 x float> @llvm.sshl.sat.v4i32(<4 x i32>, <4 x i32>)
