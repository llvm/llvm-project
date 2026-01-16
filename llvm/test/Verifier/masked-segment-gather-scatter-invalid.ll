; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

define <4 x i32> @gather_fixed_data(<vscale x 1 x ptr> %ptrs, <vscale x 1 x i1> %mask, <4 x i32> %passthru) {
; CHECK: masked_segment_gather: data type must be a scalable vector
  %res = call <4 x i32> @llvm.masked.segment.gather.v4i32.nxv1p0(<vscale x 1 x ptr> align 4 %ptrs, <vscale x 1 x i1> %mask, <4 x i32> %passthru)
  ret <4 x i32> %res
}

define <vscale x 4 x i32> @gather_wide_ptrs(<vscale x 4 x ptr> %ptrs, <vscale x 1 x i1> %mask, <vscale x 4 x i32> %passthru) {
; CHECK: masked_segment_gather: pointer type must be <vscale x 1 x ptr>
  %res = call <vscale x 4 x i32> @llvm.masked.segment.gather.nxv4i32.nxv4p0(<vscale x 4 x ptr> align 4 %ptrs, <vscale x 1 x i1> %mask, <vscale x 4 x i32> %passthru)
  ret <vscale x 4 x i32> %res
}

define <vscale x 4 x i32> @gather_non_ptrs(<vscale x 1 x i32> %ptrs, <vscale x 1 x i1> %mask, <vscale x 4 x i32> %passthru) {
; CHECK: masked_segment_gather: pointer vector elements must be pointers
  %res = call <vscale x 4 x i32> @llvm.masked.segment.gather.nxv4i32.nxv1i32(<vscale x 1 x i32> %ptrs, <vscale x 1 x i1> %mask, <vscale x 4 x i32> %passthru)
  ret <vscale x 4 x i32> %res
}

define void @scatter_fixed_data(<4 x i32> %value, <vscale x 1 x ptr> %ptrs, <vscale x 1 x i1> %mask) {
; CHECK: masked_segment_scatter: data type must be a scalable vector
  call void @llvm.masked.segment.scatter.v4i32.nxv1p0(<4 x i32> %value, <vscale x 1 x ptr> align 4 %ptrs, <vscale x 1 x i1> %mask)
  ret void
}

define void @scatter_wide_ptrs(<vscale x 4 x i32> %value, <vscale x 4 x ptr> %ptrs, <vscale x 1 x i1> %mask) {
; CHECK: masked_segment_scatter: pointer type must be <vscale x 1 x ptr>
  call void @llvm.masked.segment.scatter.nxv4i32.nxv4p0(<vscale x 4 x i32> %value, <vscale x 4 x ptr> align 4 %ptrs, <vscale x 1 x i1> %mask)
  ret void
}

define void @scatter_non_ptrs(<vscale x 4 x i32> %value, <vscale x 1 x i32> %ptrs, <vscale x 1 x i1> %mask) {
; CHECK: masked_segment_scatter: pointer vector elements must be pointers
  call void @llvm.masked.segment.scatter.nxv4i32.nxv1i32(<vscale x 4 x i32> %value, <vscale x 1 x i32> %ptrs, <vscale x 1 x i1> %mask)
  ret void
}
