; RUN: not opt -S -passes=verify -disable-output < %s 2>&1 | FileCheck %s

; Reject stepvector intrinsics that return a scalar
; CHECK: intrinsic return type (overload type 0) expected any vector type, but got i32
; CHECK-NEXT: declare i32 @llvm.stepvector.i32()
declare i32 @llvm.stepvector.i32()

; Reject vectors with non-integer elements
; CHECK: stepvector only supported for vectors of integers with a bitwidth of at least 8
; CHECK-NEXT: call <vscale x 4 x float> @llvm.stepvector.nxv4f32()
declare <vscale x 4 x float> @llvm.stepvector.nxv4f32()
define <vscale x 4 x float> @stepvector_float() {
  %1 = call <vscale x 4 x float> @llvm.stepvector.nxv4f32()
  ret <vscale x 4 x float> %1
}

; Reject vectors of integers less than 8 bits in width
; CHECK: stepvector only supported for vectors of integers with a bitwidth of at least 8
; CHECK-NEXT: call <vscale x 16 x i1> @llvm.stepvector.nxv16i1()
declare <vscale x 16 x i1> @llvm.stepvector.nxv16i1()
define <vscale x 16 x i1> @stepvector_i1() {
  %1 = call <vscale x 16 x i1> @llvm.stepvector.nxv16i1()
  ret <vscale x 16 x i1> %1
}
