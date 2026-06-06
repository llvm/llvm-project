; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-as %s -o - | llvm-dis | FileCheck %s

define <8 x i32> @interleave_fixed(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: @interleave_fixed
; CHECK: %res = call <8 x i32> @llvm.vector.interleave2.v8i32(<4 x i32> %a, <4 x i32> %b)

  %res = call <8 x i32> @llvm.experimental.vector.interleave2.v8i32(<4 x i32> %a, <4 x i32> %b)
  ret <8 x i32> %res
}

define { <4 x i32>, <4 x i32> } @deinterleave_fixed(<8 x i32> %a) {
; CHECK-LABEL: @deinterleave_fixed
; CHECK: %res = call { <4 x i32>, <4 x i32> } @llvm.vector.deinterleave2.v8i32(<8 x i32> %a)

  %res = call { <4 x i32>, <4 x i32> } @llvm.experimental.vector.deinterleave2.v8i32(<8 x i32> %a)
  ret { <4 x i32>, <4 x i32> } %res
}

define <vscale x 8 x i32> @interleave_scalable(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: @interleave_scalable
; CHECK: %res = call <vscale x 8 x i32> @llvm.vector.interleave2.nxv8i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)

  %res = call <vscale x 8 x i32> @llvm.experimental.vector.interleave2.nxv8i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b)
  ret <vscale x 8 x i32> %res
}

define { <vscale x 4 x i32>, <vscale x 4 x i32> } @deinterleave_scalable(<vscale x 8 x i32> %a) {
; CHECK-LABEL: @deinterleave_scalable
; CHECK: %res = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.vector.deinterleave2.nxv8i32(<vscale x 8 x i32> %a)

  %res = call { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.experimental.vector.deinterleave2.nxv8i32(<vscale x 8 x i32> %a)
  ret { <vscale x 4 x i32>, <vscale x 4 x i32> } %res
}

declare <8 x i32> @llvm.experimental.vector.interleave2.v8i32(<4 x i32>, <4 x i32>)
; CHECK: <8 x i32> @llvm.vector.interleave2.v8i32(<4 x i32>, <4 x i32>)

declare  { <4 x i32>, <4 x i32> } @llvm.experimental.vector.deinterleave2.v8i32(<8 x i32>)
; CHECK: declare  { <4 x i32>, <4 x i32> } @llvm.vector.deinterleave2.v8i32(<8 x i32>)

declare <vscale x 8 x i32> @llvm.experimental.vector.interleave2.nxv8i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
; CHECK: <vscale x 8 x i32> @llvm.vector.interleave2.nxv8i32(<vscale x 4 x i32>, <vscale x 4 x i32>)

declare  { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.experimental.vector.deinterleave2.nxv8i32(<vscale x 8 x i32>)
; CHECK: declare  { <vscale x 4 x i32>, <vscale x 4 x i32> } @llvm.vector.deinterleave2.nxv8i32(<vscale x 8 x i32>)
