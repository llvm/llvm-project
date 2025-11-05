; RUN: not llc -mtriple=x86_64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; Test that using MMX register constraint 'y' (64-bit) with a 256-bit vector
; produces a proper error message instead of an assertion failure.

; CHECK: error: couldn't allocate output register for constraint 'y'

define <8 x i32> @test_mmx_constraint_size_mismatch() {
entry:
  %out = tail call <8 x i32> asm "something $0", "=y"()
  ret <8 x i32> %out
}

; Also test with a different vector size
define <4 x i32> @test_mmx_constraint_128bit() {
entry:
  %out = tail call <4 x i32> asm "something $0", "=y"()
  ret <4 x i32> %out
}

