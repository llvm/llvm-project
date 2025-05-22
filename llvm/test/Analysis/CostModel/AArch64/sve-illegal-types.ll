; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu -mattr=+sve  < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void @load_store(ptr %ptrs) {
; CHECK-LABEL: 'load_store'
; CHECK-NEXT: Invalid cost for instruction: %load1 = load <vscale x 1 x i128>, ptr undef
; CHECK-NEXT: Invalid cost for instruction: %load2 = load <vscale x 2 x i128>, ptr undef
; CHECK-NEXT: Invalid cost for instruction: %load3 = load <vscale x 1 x fp128>, ptr undef
; CHECK-NEXT: Invalid cost for instruction: %load4 = load <vscale x 2 x fp128>, ptr undef
; CHECK-NEXT: Invalid cost for instruction: store <vscale x 1 x i128> %load1, ptr %ptrs
  %load1 = load <vscale x 1 x i128>, ptr undef
  %load2 = load <vscale x 2 x i128>, ptr undef
  %load3 = load <vscale x 1 x fp128>, ptr undef
  %load4 = load <vscale x 2 x fp128>, ptr undef
  store <vscale x 1 x i128> %load1, ptr %ptrs
  ret void
}

define void @masked_load_store(ptr %ptrs, ptr %val, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru) {
; CHECK-LABEL: 'masked_load_store'
; CHECK-NEXT: Invalid cost for instruction: %mload = call <vscale x 1 x i128> @llvm.masked.load.nxv1i128.p0(ptr %val, i32 8, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru)
; CHECK-NEXT: Invalid cost for instruction: call void @llvm.masked.store.nxv1i128.p0(<vscale x 1 x i128> %mload, ptr %ptrs, i32 8, <vscale x 1 x i1> %mask)
  %mload = call <vscale x 1 x i128> @llvm.masked.load.nxv1i128(ptr %val, i32 8, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru)
  call void @llvm.masked.store.nxv1i128(<vscale x 1 x i128> %mload, ptr %ptrs, i32 8, <vscale x 1 x i1> %mask)
  ret void
}

define void @masked_gather_scatter(<vscale x 1 x ptr> %ptrs, <vscale x 1 x ptr> %val, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru) {
; CHECK-LABEL: 'masked_gather_scatter'
; CHECK-NEXT: Invalid cost for instruction: %mgather = call <vscale x 1 x i128> @llvm.masked.gather.nxv1i128.nxv1p0(<vscale x 1 x ptr> %val, i32 0, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru)
; CHECK-NEXT: Invalid cost for instruction: call void @llvm.masked.scatter.nxv1i128.nxv1p0(<vscale x 1 x i128> %mgather, <vscale x 1 x ptr> %ptrs, i32 0, <vscale x 1 x i1> %mask)
  %mgather = call <vscale x 1 x i128> @llvm.masked.gather.nxv1i128(<vscale x 1 x ptr> %val, i32 0, <vscale x 1 x i1> %mask, <vscale x 1 x i128> %passthru)
  call void @llvm.masked.scatter.nxv1i128(<vscale x 1 x i128> %mgather, <vscale x 1 x ptr> %ptrs, i32 0, <vscale x 1 x i1> %mask)
  ret void
}

declare <vscale x 1 x i128> @llvm.masked.load.nxv1i128(ptr, i32, <vscale x 1 x i1>, <vscale x 1 x i128>)
declare <vscale x 1 x i128> @llvm.masked.gather.nxv1i128(<vscale x 1 x ptr>, i32, <vscale x 1 x i1>, <vscale x 1 x i128>)

declare void @llvm.masked.store.nxv1i128(<vscale x 1 x i128>, ptr, i32, <vscale x 1 x i1>)
declare void @llvm.masked.scatter.nxv1i128(<vscale x 1 x i128>, <vscale x 1 x ptr>, i32, <vscale x 1 x i1>)
