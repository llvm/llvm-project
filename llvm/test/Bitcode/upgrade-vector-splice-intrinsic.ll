; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

define <8 x half> @splice_fixed_left(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: @splice_fixed_left
; CHECK: %1 = call <8 x half> @llvm.vector.splice.left.v8f16(<8 x half> %a, <8 x half> %b, i32 2)

  %res = call <8 x half> @llvm.experimental.vector.splice.v8f16(<8 x half> %a, <8 x half> %b, i32 2)
  ret <8 x half> %res
}

define <8 x half> @splice_fixed_right(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: @splice_fixed_right
; CHECK: %1 = call <8 x half> @llvm.vector.splice.right.v8f16(<8 x half> %a, <8 x half> %b, i32 2)

  %res = call <8 x half> @llvm.experimental.vector.splice.v8f16(<8 x half> %a, <8 x half> %b, i32 -2)
  ret <8 x half> %res
}

define <vscale x 8 x half> @splice_scalable(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: @splice_scalable
; CHECK: %1 = call <vscale x 8 x half> @llvm.vector.splice.left.nxv8f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b, i32 2)

  %res = call <vscale x 8 x half> @llvm.experimental.vector.splice.nxv8f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b, i32 2)
  ret <vscale x 8 x half> %res
}

declare <8 x half> @llvm.experimental.vector.splice.v8f16(<8 x half>, <8 x half>, i32 immarg)
; CHECK: declare <8 x half> @llvm.vector.splice.left.v8f16(<8 x half>, <8 x half>, i32 immarg)

declare <vscale x 8 x half> @llvm.experimental.vector.splice.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, i32 immarg)
; CHECK: declare <vscale x 8 x half> @llvm.vector.splice.left.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, i32 immarg)
