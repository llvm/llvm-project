; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

define <8 x half> @splice_fixed(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: @splice_fixed
; CHECK: %res = call <8 x half> @llvm.vector.splice.v8f16(<8 x half> %a, <8 x half> %b, i32 2)

  %res = call <8 x half> @llvm.experimental.vector.splice.v8f16(<8 x half> %a, <8 x half> %b, i32 2)
  ret <8 x half> %res
}

define <vscale x 8 x half> @splice_scalable(<vscale x 8 x half> %a, <vscale x 8 x half> %b) {
; CHECK-LABEL: @splice_scalable
; CHECK: %res = call <vscale x 8 x half> @llvm.vector.splice.nxv8f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b, i32 2)

  %res = call <vscale x 8 x half> @llvm.experimental.vector.splice.nxv8f16(<vscale x 8 x half> %a, <vscale x 8 x half> %b, i32 2)
  ret <vscale x 8 x half> %res
}

declare <8 x half> @llvm.experimental.vector.splice.v8f16(<8 x half>, <8 x half>, i32 immarg)
; CHECK: declare <8 x half> @llvm.vector.splice.v8f16(<8 x half>, <8 x half>, i32 immarg)

declare <vscale x 8 x half> @llvm.experimental.vector.splice.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, i32 immarg)
; CHECK: declare <vscale x 8 x half> @llvm.vector.splice.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>, i32 immarg)
