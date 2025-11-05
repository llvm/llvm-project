; RUN: opt -mtriple=aarch64-unknown-linux-gnu -mattr=+sve -passes=vector-combine %s -S -o - | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define <vscale x 4 x i16> @interleave2_same_const_splat_nxv4i16() {
;CHECK-LABEL: @interleave2_same_const_splat_nxv4i16(
;CHECK: ret <vscale x 4 x i16> bitcast (<vscale x 2 x i32> splat (i32 196611) to <vscale x 4 x i16>)
  %retval = call <vscale x 4 x i16> @llvm.vector.interleave2.nxv4i16(<vscale x 2 x i16> splat(i16 3), <vscale x 2 x i16> splat(i16 3))
  ret <vscale x 4 x i16> %retval
}
