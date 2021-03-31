; RUN: opt  -instsimplify -S < %s 2>%t | FileCheck %s

; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; Test back to back reverse shuffles are eliminated.
define <vscale x 4 x i32> @shuffle_b2b_reverse(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @shuffle_b2b_reverse(
; CHECK: ret <vscale x 4 x i32> %a
  %rev = tail call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> %a)
  %rev.rev = tail call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> %rev)
  ret <vscale x 4 x i32> %rev.rev
}

declare <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32>)
