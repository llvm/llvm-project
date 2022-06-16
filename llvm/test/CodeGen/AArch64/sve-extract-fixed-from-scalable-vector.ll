; RUN: not --crash llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; Extracting a fixed-length vector from an illegal subvector

; CHECK-ERROR: ERROR: Extracting a fixed-length vector from an illegal scalable vector is not yet supported
define <4 x i32> @extract_v4i32_nxv16i32_12(<vscale x 16 x i32> %arg) {
  %ext = call <4 x i32> @llvm.vector.extract.v4i32.nxv16i32(<vscale x 16 x i32> %arg, i64 12)
  ret <4 x i32> %ext
}

declare <4 x i32> @llvm.vector.extract.v4i32.nxv16i32(<vscale x 16 x i32>, i64)
