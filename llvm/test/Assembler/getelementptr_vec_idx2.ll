; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test that a vector pointer may be used with a scalar index.
; Test that a vector pointer and vector index should have the same vector width

; This code is correct
define <2 x ptr> @test2(<2 x ptr> %a) {
  %w = getelementptr i32, <2 x ptr> %a, i32 2
  ret <2 x ptr> %w
}

; This code is correct
define <2 x ptr> @test3(ptr %a) {
  %w = getelementptr i32, ptr %a, <2 x i32> <i32 2, i32 2>
  ret <2 x ptr> %w
}

; CHECK: getelementptr vector index has a wrong number of elements

define <2 x i32> @test1(<2 x ptr> %a) {
  %w = getelementptr i32, <2 x ptr> %a, <4 x i32><i32 2, i32 2, i32 2, i32 2>
  ret <2 x i32> %w
}

