; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this doesn't end up being an entirely perfect shuffle.
; CHECK: vshuff
; CHECK-NOT: vdeal
define <32 x i32> @f0(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %v0 = shufflevector <32 x i32> %a0, <32 x i32> %a1, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <32 x i32> %v0
}

attributes #0 = { nounwind memory(none) "target-features"="+hvxv62,+hvx-length128b" }
