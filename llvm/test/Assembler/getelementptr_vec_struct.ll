; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test that a vector struct index with non-equal elements is rejected.

; CHECK: invalid getelementptr indices

define <2 x ptr> @test7(<2 x ptr> %a) {
  %w = getelementptr {i32, i32}, <2 x ptr> %a, <2 x i32> <i32 5, i32 9>, <2 x i32> <i32 0, i32 1>
  ret <2 x ptr> %w
}
