; RUN: llc -mtriple=aarch64 -mattr=+neon < %s | FileCheck %s

define <16 x i8> @test_zero_splat(<16 x i8> %tbl, <16 x i8> %idx) {
; CHECK-LABEL: test_zero_splat:
; CHECK: tbl
; CHECK: ret
  %res = call <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8> zeroinitializer, <16 x i8> %tbl, <16 x i8> %idx)
  ret <16 x i8> %res
}

declare <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)