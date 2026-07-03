; RUN: not --crash llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Do not know how to widen this operator's operand!

; TODO: Support broadcasts from 32-bit vec
define <vscale x 8 x half> @broadcast_single_f16(<2 x half> %a) {
  %out = call <vscale x 8 x half> @llvm.vector.broadcast.nxv8f16(<2 x half> %a)
  ret <vscale x 8 x half> %out
}
