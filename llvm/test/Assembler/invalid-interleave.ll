; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Input element count without vscale is not a multiple of 3.

; CHECK: error: intrinsic return struct element 0 is 1/nth (n=3) elements vector of overload type 0, so overload type 0 expected vector with multiple of 3 elements, but got <vscale x 8 x i8>
define { <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32> } @test(<vscale x 8 x i8> %ptr) {
  %v = tail call { <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32> } @llvm.vector.deinterleave3.nxv6i32(<vscale x 8 x i8> %ptr)
  ret { <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32> } %v
}
