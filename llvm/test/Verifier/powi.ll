; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; The exponent of llvm.powi is scalar even when the base is a vector.
; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare <4 x float> @llvm.powi.v4f32.v4i32(<4 x float>, <4 x i32>)
declare <4 x float> @llvm.powi.v4f32.v4i32(<4 x float>, <4 x i32>)
