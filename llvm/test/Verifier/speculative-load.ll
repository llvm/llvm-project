; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare <3 x i32> @llvm.speculative.load.v3i32.p0(ptr, i64)
declare <vscale x 3 x i32> @llvm.speculative.load.nxv3i32.p0(ptr, i64)

define <3 x i32> @test_non_power_of_2_fixed(ptr %ptr) {
; CHECK: llvm.speculative.load type must have a power-of-2 size
; CHECK-NEXT: %res = call <3 x i32> @llvm.speculative.load.v3i32.p0(ptr %ptr, i64 0)
  %res = call <3 x i32> @llvm.speculative.load.v3i32.p0(ptr %ptr, i64 0)
  ret <3 x i32> %res
}

define <vscale x 3 x i32> @test_non_power_of_2_scalable(ptr %ptr) {
; CHECK: llvm.speculative.load type must have a power-of-2 size
; CHECK-NEXT: %res = call <vscale x 3 x i32> @llvm.speculative.load.nxv3i32.p0(ptr %ptr, i64 0)
  %res = call <vscale x 3 x i32> @llvm.speculative.load.nxv3i32.p0(ptr %ptr, i64 0)
  ret <vscale x 3 x i32> %res
}
