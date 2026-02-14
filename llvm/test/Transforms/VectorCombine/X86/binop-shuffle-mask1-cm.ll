; REQUIRES: asserts
; RUN: opt -passes=vector-combine -debug-only=vector-combine -disable-output < %s 2>&1 | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; Test that the cost modeling for folding shuffled binops uses the correct
; values for the masks of each operand.

; CHECK: Found a shuffle feeding a shuffled binop:   [[SHUF:%.*]] = shufflevector <4 x i8> [[PREV_SHUF:%.*]], <4 x i8> poison, <8 x i32> <i32 0, i32 poison, i32 1, i32 poison, i32 2, i32 poison, i32 3, i32 poison>
; CHECK-NEXT:  OldCost: 3 vs NewCost: 5


define <8 x i8> @test(<4 x i8> %0) #0 {
  %2 = shufflevector <4 x i8> %0, <4 x i8> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %3 = xor <4 x i8> %0, %2
  %4 = shufflevector <4 x i8> %3, <4 x i8> poison, <8 x i32> <i32 0, i32 poison, i32 1, i32 poison, i32 2, i32 poison, i32 3, i32 poison>
  ret <8 x i8> %4
}

attributes #0 = { "target-cpu"="core-avx2" }
