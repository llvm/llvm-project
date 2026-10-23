; Offsets that do not fit into a single SVE register are rejected, as for scatters.
; RUN: not --crash llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Do not know how to split this operator's operand!

declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.nxv4i32(<vscale x 4 x i1>, ptr, <vscale x 4 x i64>)

define <vscale x 4 x i32> @f(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i64> %offsets) {
  %l = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.nxv4i32(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i64> %offsets)
  ret <vscale x 4 x i32> %l
}
