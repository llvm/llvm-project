; RUN: opt  -instsimplify -S < %s | FileCheck %s

; Test back to back reverse shuffles are eliminated.
define <vscale x 4 x i32> @shuffle_b2b_reverse(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @shuffle_b2b_reverse(
; CHECK: ret <vscale x 4 x i32> %a
  %rev = tail call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> %a)
  %rev.rev = tail call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> %rev)
  ret <vscale x 4 x i32> %rev.rev
}

declare <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32>)
