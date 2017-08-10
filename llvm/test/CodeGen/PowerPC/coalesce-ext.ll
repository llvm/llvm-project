; RUN: llc -verify-machineinstrs -mcpu=g5 -mtriple=powerpc64-apple-darwin < %s | FileCheck %s
; Check that the peephole optimizer knows about sext and zext instructions.
; CHECK: test1sext
define i32 @test1sext(i64 %A, i64 %B, i32* %P, i64 *%P2) nounwind {
  %C = add i64 %A, %B
  ; CHECK: add [[SUM:r[0-9]+]], r3, r4
  %D = trunc i64 %C to i32
  %E = shl i64 %C, 32
  %F = ashr i64 %E, 32
  ; CHECK: extsw [[EXT:r[0-9]+]], [[SUM]]
  store volatile i64 %F, i64 *%P2
  ; CHECK: std [[EXT]]
  store volatile i32 %D, i32* %P
  ; Reuse low bits of extended register, don't extend live range of SUM.
  ; CHECK: stw [[EXT]]
  %R = add i32 %D, %D
  ret i32 %R
}
