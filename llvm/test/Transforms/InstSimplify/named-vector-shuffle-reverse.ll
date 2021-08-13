; RUN: opt  -instsimplify -S < %s | FileCheck %s

; Test back to back reverse shuffles are eliminated.
define <vscale x 4 x i32> @shuffle_b2b_reverse(<vscale x 4 x i32> %a) {
; CHECK-LABEL: @shuffle_b2b_reverse(
; CHECK: ret <vscale x 4 x i32> %a
  %rev = tail call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> %a)
  %rev.rev = tail call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> %rev)
  ret <vscale x 4 x i32> %rev.rev
}

; Test reverse of a splat is eliminated.
define <vscale x 4 x i32> @splat_reverse(i32 %a) {
; CHECK-LABEL: @splat_reverse(
; CHECK-NEXT:    [[SPLAT_INSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[A:%.*]], i32 0
; CHECK-NEXT:    [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLAT_INSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    ret <vscale x 4 x i32> [[SPLAT]]
;
  %splat_insert = insertelement <vscale x 4 x i32> poison, i32 %a, i32 0
  %splat = shufflevector <vscale x 4 x i32> %splat_insert, <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
  %rev = tail call <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32> %splat)
  ret <vscale x 4 x i32> %rev
}

declare <vscale x 4 x i32> @llvm.experimental.vector.reverse.nxv4i32(<vscale x 4 x i32>)
