; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

define <4 x i32> @stepvector_fixed() {
; CHECK-LABEL: @stepvector_fixed
; CHECK: %res = call <4 x i32> @llvm.stepvector.v4i32()

  %res = call <4 x i32> @llvm.experimental.stepvector.v4i32()
  ret <4 x i32> %res
}

define <vscale x 4 x i32> @stepvector_scalable() {
; CHECK-LABEL: @stepvector_scalable
; CHECK: %res = call <vscale x 4 x i32> @llvm.stepvector.nxv4i32()

  %res = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
  ret <vscale x 4 x i32> %res
}


declare <4 x i32> @llvm.experimental.stepvector.v4i32()
; CHECK: <4 x i32> @llvm.stepvector.v4i32()

declare <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; CHECK: <vscale x 4 x i32> @llvm.stepvector.nxv4i32()

