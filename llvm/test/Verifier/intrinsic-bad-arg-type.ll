; RUN: not opt -S -passes=verify 2>&1 < %s | FileCheck %s

; CHECK: Intrinsic has incorrect argument type!
; CHECK-NEXT: ptr @llvm.masked.load.nxv4i32.p0

define <vscale x 4 x i32> @masked_load(ptr %addr, <4 x i1> %mask, <vscale x 4 x i32> %dst) {
  %res = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr %addr, i32 4, <4 x i1> %mask, <vscale x 4 x i32> %dst)
  ret <vscale x 4 x i32> %res
}
declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, i32, <4 x i1>, <vscale x 4 x i32>)
