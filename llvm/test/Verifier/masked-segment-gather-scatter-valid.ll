; RUN: opt -passes=verify -S < %s | FileCheck %s

define <vscale x 4 x i32> @valid_gather(<vscale x 1 x ptr> %ptrs, <vscale x 1 x i1> %mask, <vscale x 4 x i32> %passthru) {
; CHECK-LABEL: @valid_gather(
; CHECK: call <vscale x 4 x i32> @llvm.masked.segment.gather.nxv4i32.nxv1p0
  %res = call <vscale x 4 x i32> @llvm.masked.segment.gather.nxv4i32.nxv1p0(<vscale x 1 x ptr> align 4 %ptrs, <vscale x 1 x i1> %mask, <vscale x 4 x i32> %passthru)
  ret <vscale x 4 x i32> %res
}

define void @valid_scatter(<vscale x 4 x i32> %value, <vscale x 1 x ptr> %ptrs, <vscale x 1 x i1> %mask) {
; CHECK-LABEL: @valid_scatter(
; CHECK: call void @llvm.masked.segment.scatter.nxv4i32.nxv1p0
  call void @llvm.masked.segment.scatter.nxv4i32.nxv1p0(<vscale x 4 x i32> %value, <vscale x 1 x ptr> align 4 %ptrs, <vscale x 1 x i1> %mask)
  ret void
}
