; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu -mattr=+sve  < %s | FileCheck %s

define void @load_store(<vscale x 1 x i128>* %ptrs) {
; CHECK-LABEL: 'load_store'
; CHECK-NEXT: Invalid cost for instruction: %load1 = load <vscale x 1 x i128>, <vscale x 1 x i128>* undef
; CHECK-NEXT: Invalid cost for instruction: %load2 = load <vscale x 2 x i128>, <vscale x 2 x i128>* undef
; CHECK-NEXT: Invalid cost for instruction: %load3 = load <vscale x 1 x fp128>, <vscale x 1 x fp128>* undef
; CHECK-NEXT: Invalid cost for instruction: %load4 = load <vscale x 2 x fp128>, <vscale x 2 x fp128>* undef
; CHECK-NEXT: Invalid cost for instruction: store <vscale x 1 x i128> %load1, <vscale x 1 x i128>* %ptrs
  %load1 = load <vscale x 1 x i128>, <vscale x 1 x i128>* undef
  %load2 = load <vscale x 2 x i128>, <vscale x 2 x i128>* undef
  %load3 = load <vscale x 1 x fp128>, <vscale x 1 x fp128>* undef
  %load4 = load <vscale x 2 x fp128>, <vscale x 2 x fp128>* undef
  store <vscale x 1 x i128> %load1, <vscale x 1 x i128>* %ptrs
  ret void
}

define void @masked_load_store(<vscale x 1 x i128>* %ptrs, <vscale x 1 x i128>* %val, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru) {
; CHECK-LABEL: 'masked_load_store'
; CHECK-NEXT: Invalid cost for instruction: %mload = call <vscale x 1 x i128> @llvm.masked.load.nxv1i128.p0nxv1i128(<vscale x 1 x i128>* %val, i32 8, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru)
; CHECK-NEXT: Invalid cost for instruction: call void @llvm.masked.store.nxv1i128.p0nxv1i128(<vscale x 1 x i128> %mload, <vscale x 1 x i128>* %ptrs, i32 8, <vscale x 1 x i1> %mask)
  %mload = call <vscale x 1 x i128> @llvm.masked.load.nxv1i128(<vscale x 1 x i128>* %val, i32 8, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru)
  call void @llvm.masked.store.nxv1i128(<vscale x 1 x i128> %mload, <vscale x 1 x i128>* %ptrs, i32 8, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_gather_scatter(<vscale x 1 x i128*> %ptrs, <vscale x 1 x i128*> %val, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru) {
; CHECK-LABEL: 'masked_gather_scatter'
; CHECK-NEXT: Invalid cost for instruction: %mgather = call <vscale x 1 x i128> @llvm.masked.gather.nxv1i128.nxv1p0i128(<vscale x 1 x i128*> %val, i32 0, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru)
; CHECK-NEXT: Invalid cost for instruction: call void @llvm.masked.scatter.nxv1i128.nxv1p0i128(<vscale x 1 x i128> %mgather, <vscale x 1 x i128*> %ptrs, i32 0, <vscale x 1 x i1> %mask)
  %mgather = call <vscale x 1 x i128> @llvm.masked.gather.nxv1i128(<vscale x 1 x i128*> %val, i32 0, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru)
  call void @llvm.masked.scatter.nxv1i128(<vscale x 1 x i128> %mgather, <vscale x 1 x i128*> %ptrs, i32 0, <vscale x 1 x i1> %mask)
  ret void
}

declare <vscale x 1 x i128> @llvm.masked.load.nxv1i128(<vscale x 1 x i128>*, i32, <vscale x 1 x i1>, <vscale x 1 x i128>)
declare <vscale x 1 x i128> @llvm.masked.gather.nxv1i128(<vscale x 1 x i128*>, i32, <vscale x 1 x i1>, <vscale x 1 x i128>)

declare void @llvm.masked.store.nxv1i128(<vscale x 1 x i128>, <vscale x 1 x i128>*, i32, <vscale x 1 x i1>)
declare void @llvm.masked.scatter.nxv1i128(<vscale x 1 x i128>, <vscale x 1 x i128*>, i32, <vscale x 1 x i1>)
