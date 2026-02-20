; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: invalid intrinsic signature

define <2 x i1> @test(<2 x i32> %b) {
  %2 = tail call <2 x i1> @llvm.aarch64.neon.smull.v2i1(<2 x i32> %b, <2 x i32> %b)
  ret <2 x i1> %2
}
