; RUN: opt < %s -cost-model -analyze -mtriple=aarch64-linux-gnu -mattr=+sve | FileCheck %s

define <16 x i32> @extract_cost(<vscale x 4 x i32> %vec) {
; CHECK-LABEL: 'extract_cost'
; CHECK-NEXT: Cost Model: Found an estimated cost of 72 for instruction:   %ret = call <16 x i32> @llvm.experimental.vector.extract.v16i32.nxv4i32(<vscale x 4 x i32> %vec, i64 0)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret <16 x i32> %ret

  %ret = call <16 x i32> @llvm.experimental.vector.extract.v16i32.nxv4i32(<vscale x 4 x i32> %vec, i64 0)
  ret <16 x i32> %ret
}

define <vscale x 4 x i32> @insert_cost(<vscale x 4 x i32> %vec, <16 x i32> %subVec) {
; CHECK-LABEL: 'insert_cost'
; CHECK-NEXT: Cost Model: Found an estimated cost of 72 for instruction:   %ret = call <vscale x 4 x i32> @llvm.experimental.vector.insert.nxv4i32.v16i32(<vscale x 4 x i32> %vec, <16 x i32> %subVec, i64 0)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 4 x i32> %ret

  %ret = call <vscale x 4 x i32> @llvm.experimental.vector.insert.nxv4i32.v16i32(<vscale x 4 x i32> %vec, <16 x i32> %subVec, i64 0)
  ret <vscale x 4 x i32> %ret
}

define <vscale x 4 x i32> @extract_cost_scalable(<vscale x 16 x i32> %vec) {
; CHECK-LABEL: 'extract_cost_scalable'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %ret = call <vscale x 4 x i32> @llvm.experimental.vector.extract.nxv4i32.nxv16i32(<vscale x 16 x i32> %vec, i64 0)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 4 x i32> %ret

  %ret = call <vscale x 4 x i32> @llvm.experimental.vector.extract.nxv4i32.nxv16i32(<vscale x 16 x i32> %vec, i64 0)
  ret <vscale x 4 x i32> %ret
}

define <vscale x 16 x i32> @insert_cost_scalable(<vscale x 16 x i32> %vec, <vscale x 4 x i32> %subVec) {
; CHECK-LABEL: 'insert_cost_scalable'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %ret = call <vscale x 16 x i32> @llvm.experimental.vector.insert.nxv16i32.nxv4i32(<vscale x 16 x i32> %vec, <vscale x 4 x i32> %subVec, i64 0)
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   ret <vscale x 16 x i32> %ret

  %ret = call <vscale x 16 x i32> @llvm.experimental.vector.insert.nxv16i32.nxv4i32(<vscale x 16 x i32> %vec, <vscale x 4 x i32> %subVec, i64 0)
  ret <vscale x 16 x i32> %ret
}

declare <16 x i32> @llvm.experimental.vector.extract.v16i32.nxv4i32(<vscale x 4 x i32>, i64)
declare <vscale x 4 x i32> @llvm.experimental.vector.insert.nxv4i32.v16i32(<vscale x 4 x i32>, <16 x i32>, i64)
declare <vscale x 4 x i32> @llvm.experimental.vector.extract.nxv4i32.nxv16i32(<vscale x 16 x i32>, i64)
declare <vscale x 16 x i32> @llvm.experimental.vector.insert.nxv16i32.nxv4i32(<vscale x 16 x i32>, <vscale x 4 x i32>, i64)
