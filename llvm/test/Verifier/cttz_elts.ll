; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; The return type must be a scalar integer.
; CHECK: intrinsic return type (overload type 0) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.experimental.cttz.elts.v4i32.v4i32(<4 x i32>, i1)
declare <4 x i32> @llvm.experimental.cttz.elts.v4i32.v4i32(<4 x i32>, i1)

; The first argument must be a vector of integers.
; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.experimental.cttz.elts.i32.i32(i32, i1)
declare i32 @llvm.experimental.cttz.elts.i32.i32(i32, i1)

; CHECK: intrinsic argument 0 type (overload type 1) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare i32 @llvm.experimental.cttz.elts.i32.v4f32(<4 x float>, i1)
declare i32 @llvm.experimental.cttz.elts.i32.v4f32(<4 x float>, i1)
