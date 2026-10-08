; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx_neg3(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 -3, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <vscale x 2 x double> @splice_nxv2f64_idx_neg3_vscale_min1(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <vscale x 2 x double> @llvm.experimental.vp.splice.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 -3, <vscale x 2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <vscale x 2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <vscale x 2 x double> @splice_nxv2f64_idx_neg5_vscale_min2(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 %evl1, i32 %evl2) #1 {
  %res = call <vscale x 2 x double> @llvm.experimental.vp.splice.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 -5, <vscale x 2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <vscale x 2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx2(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 2, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx3(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #1 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 4, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

attributes #0 = { vscale_range(1,16) }
attributes #1 = { vscale_range(2,16) }

; The vector operands of integer vp reductions must have integer elements.
; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.add.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.add.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.mul.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.mul.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.and.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.and.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.or.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.or.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.xor.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.xor.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.smax.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.smax.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.smin.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.smin.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.umax.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.umax.v4f32(float, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vp.reduce.umin.v4f32(float, <4 x float>, <4 x i1>, i32)
declare float @llvm.vp.reduce.umin.v4f32(float, <4 x float>, <4 x i1>, i32)

; The vector operands of floating-point vp reductions must have floating-point elements.
; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fadd.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fadd.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fmul.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fmul.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fmax.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fmax.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fmin.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fmin.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fmaximum.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fmaximum.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vp.reduce.fminimum.v4i32(i32, <4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vp.reduce.fminimum.v4i32(i32, <4 x i32>, <4 x i1>, i32)

; The integer vp division and remainder intrinsics must return a vector of integers.
; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vp.sdiv.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)
declare <4 x float> @llvm.vp.sdiv.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vp.udiv.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)
declare <4 x float> @llvm.vp.udiv.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vp.srem.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)
declare <4 x float> @llvm.vp.srem.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)

; CHECK: intrinsic return type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vp.urem.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)
declare <4 x float> @llvm.vp.urem.v4f32(<4 x float>, <4 x float>, <4 x i1>, i32)

; The vp.cttz.elts result must be a scalar integer and its first argument must be a vector of integers.
; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.vp.cttz.elts.v4i32.v4i32(<4 x i32>, i1, <4 x i1>, i32)
declare <4 x i32> @llvm.vp.cttz.elts.v4i32.v4i32(<4 x i32>, i1, <4 x i1>, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.vp.cttz.elts.i32.i32(i32, i1, i1, i32)
declare i32 @llvm.vp.cttz.elts.i32.i32(i32, i1, i1, i32)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare i32 @llvm.vp.cttz.elts.i32.v4f32(<4 x float>, i1, <4 x i1>, i32)
declare i32 @llvm.vp.cttz.elts.i32.v4f32(<4 x float>, i1, <4 x i1>, i32)

; The stride argument of the strided vp memory intrinsics must be a scalar integer.
; CHECK: intrinsic argument 2 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vp.strided.store.v4f32.p0.v4i32(<4 x float>, ptr, <4 x i32>, <4 x i1>, i32)
declare void @llvm.experimental.vp.strided.store.v4f32.p0.v4i32(<4 x float>, ptr, <4 x i32>, <4 x i1>, i32)

; CHECK: intrinsic argument 1 type (overload type 2) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x float> @llvm.experimental.vp.strided.load.v4f32.p0.v4i32(ptr, <4 x i32>, <4 x i1>, i32)
declare <4 x float> @llvm.experimental.vp.strided.load.v4f32.p0.v4i32(ptr, <4 x i32>, <4 x i1>, i32)

; A scalar result is invalid, but calling it must not crash the verifier.
; CHECK: intrinsic return type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: declare i32 @llvm.experimental.vp.splice.i32(i32, i32, i32, i1, i32, i32)
declare i32 @llvm.experimental.vp.splice.i32(i32, i32, i32, i1, i32, i32)

define i32 @vp_splice_scalar(i32 %a, i32 %b, i32 %evl) {
  %r = call i32 @llvm.experimental.vp.splice.i32(i32 %a, i32 %b, i32 0, i1 true, i32 %evl, i32 %evl)
  ret i32 %r
}
