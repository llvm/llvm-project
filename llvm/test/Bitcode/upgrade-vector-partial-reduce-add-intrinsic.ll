; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

define <4 x i32> @partial_reduce_add_fixed(<16 x i32> %a) {
; CHECK-LABEL: @partial_reduce_add_fixed
; CHECK: %res = call <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v16i32(<4 x i32> zeroinitializer, <16 x i32> %a)

  %res = call <4 x i32> @llvm.experimental.vector.partial.reduce.add.v4i32.v16i32(<4 x i32> zeroinitializer, <16 x i32> %a)
  ret <4 x i32> %res
}


define <vscale x 4 x i32> @partial_reduce_add_scalable(<vscale x 16 x i32> %a) {
; CHECK-LABEL: @partial_reduce_add_scalable
; CHECK: %res = call <vscale x 4 x i32> @llvm.vector.partial.reduce.add.nxv4i32.nxv16i32(<vscale x 4 x i32> zeroinitializer, <vscale x 16 x i32> %a)

  %res = call <vscale x 4 x i32> @llvm.experimental.vector.partial.reduce.add.nxv4i32.nxv16i32(<vscale x 4 x i32> zeroinitializer, <vscale x 16 x i32> %a)
  ret <vscale x 4 x i32> %res
}

declare <4 x i32> @llvm.experimental.vector.partial.reduce.add.v4i32.v16i32(<4 x i32>, <16 x i32>)
; CHECK-DAG: declare <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v16i32(<4 x i32>, <16 x i32>)

declare <vscale x 4 x i32> @llvm.experimental.vector.partial.reduce.add.nxv4i32.nxv16i32(<vscale x 4 x i32>, <vscale x 16 x i32>)
; CHECK-DAG: declare <vscale x 4 x i32> @llvm.vector.partial.reduce.add.nxv4i32.nxv16i32(<vscale x 4 x i32>, <vscale x 16 x i32>)
