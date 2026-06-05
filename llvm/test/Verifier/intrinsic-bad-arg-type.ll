; RUN: not opt -S -passes=verify 2>&1 < %s | FileCheck %s

; CHECK: intrinsic argument 1 type (same vector width of overload type 0) expected vector with vscale x 4 elements (overload type 0 is <vscale x 4 x i32>), but got <4 x i1>
; CHECK-NEXT: ptr @llvm.masked.load.nxv4i32.p0

define <vscale x 4 x i32> @masked_load(ptr %addr, <4 x i1> %mask, <vscale x 4 x i32> %dst) {
  %res = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr %addr, <4 x i1> %mask, <vscale x 4 x i32> %dst)
  ret <vscale x 4 x i32> %res
}
declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0(ptr, <4 x i1>, <vscale x 4 x i32>)
