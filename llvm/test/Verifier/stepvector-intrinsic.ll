; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

; Reject stepvector intrinsics that return a scalar

define i32 @stepvector_i32() {
; CHECK: Intrinsic has incorrect return type!
  %1 = call i32 @llvm.stepvector.i32()
  ret i32 %1
}

; Reject vectors with non-integer elements

define <vscale x 4 x float> @stepvector_float() {
; CHECK: stepvector only supported for vectors of integers with a bitwidth of at least 8
  %1 = call <vscale x 4 x float> @llvm.stepvector.nxv4f32()
  ret <vscale x 4 x float> %1
}

; Reject vectors of integers less than 8 bits in width

define <vscale x 16 x i1> @stepvector_i1() {
; CHECK: stepvector only supported for vectors of integers with a bitwidth of at least 8
  %1 = call <vscale x 16 x i1> @llvm.stepvector.nxv16i1()
  ret <vscale x 16 x i1> %1
}

declare i32 @llvm.stepvector.i32()
declare <vscale x 4 x float> @llvm.stepvector.nxv4f32()
declare <vscale x 16 x i1> @llvm.stepvector.nxv16i1()
