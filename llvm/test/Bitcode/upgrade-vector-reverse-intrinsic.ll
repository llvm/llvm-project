; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

define <16 x i8> @reverse_fixed(<16 x i8> %a) {
; CHECK-LABEL: @reverse_fixed
; CHECK: %res = call <16 x i8> @llvm.vector.reverse.v16i8(<16 x i8> %a)

  %res = call <16 x i8> @llvm.experimental.vector.reverse.v16i8(<16 x i8> %a)
  ret <16 x i8> %res
}

define <vscale x 16 x i8> @reverse_scalable(<vscale x 16 x i8> %a) {
; CHECK-LABEL: @reverse_scalable
; CHECK: %res = call <vscale x 16 x i8> @llvm.vector.reverse.nxv16i8(<vscale x 16 x i8> %a)

  %res = call <vscale x 16 x i8> @llvm.experimental.vector.reverse.nxv16i8(<vscale x 16 x i8> %a)
  ret <vscale x 16 x i8> %res
}

declare <16 x i8> @llvm.experimental.vector.reverse.v16i8(<16 x i8>)
; CHECK: declare <16 x i8> @llvm.vector.reverse.v16i8(<16 x i8>)

declare <vscale x 16 x i8> @llvm.experimental.vector.reverse.nxv16i8(<vscale x 16 x i8>)
; CHECK: declare <vscale x 16 x i8> @llvm.vector.reverse.nxv16i8(<vscale x 16 x i8>)
