; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; The operands must be vectors of integers.
; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x i1> @llvm.experimental.vector.match.v4f32.v4f32(<4 x float>, <4 x float>, <4 x i1>)
declare <4 x i1> @llvm.experimental.vector.match.v4f32.v4f32(<4 x float>, <4 x float>, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x i1> @llvm.experimental.vector.match.v4i32.v4f32(<4 x i32>, <4 x float>, <4 x i1>)
declare <4 x i1> @llvm.experimental.vector.match.v4i32.v4f32(<4 x i32>, <4 x float>, <4 x i1>)

; The operands and the result must be vectors, and the mask must be a vector of i1.
; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got i32
; CHECK-NEXT: declare i1 @llvm.experimental.vector.match.i32.i32(i32, i32, i1)
declare i1 @llvm.experimental.vector.match.i32.i32(i32, i32, i1)

; CHECK: intrinsic argument 2 type (same vector width of overload type 0) expected vector (overload type 0 is <4 x i32>), but got i1
; CHECK-NEXT: declare <4 x i1> @llvm.experimental.vector.match.v4i32.v4i32(<4 x i32>, <4 x i32>, i1)
declare <4 x i1> @llvm.experimental.vector.match.v4i32.v4i32(<4 x i32>, <4 x i32>, i1)

; CHECK: intrinsic argument 2 vector element type expected i1, but got i32
; CHECK-NEXT: declare <8 x i1> @llvm.experimental.vector.match.v8i32.v8i32(<8 x i32>, <8 x i32>, <8 x i32>)
declare <8 x i1> @llvm.experimental.vector.match.v8i32.v8i32(<8 x i32>, <8 x i32>, <8 x i32>)
