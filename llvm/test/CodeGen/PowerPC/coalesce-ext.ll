; RUN: llc -verify-machineinstrs -mcpu=g5 -mtriple=powerpc64-unknown-linux-gnu -ppc-asm-full-reg-names < %s | FileCheck %s
; Check that the peephole optimizer knows about sext and zext instructions.
; CHECK: test1sext
define i32 @test1sext(i64 %A, i64 %B, ptr %P, ptr %P2) nounwind {
  %C = add i64 %A, %B
  ; CHECK: add [[SUM:r[0-9]+]], r3, r4
  %D = trunc i64 %C to i32
  %E = shl i64 %C, 32
  %F = ashr i64 %E, 32
  ; CHECK: extsw [[EXT:r[0-9]+]], [[SUM]]
  store volatile i64 %F, ptr %P2
  ; CHECK-DAG: std [[EXT]]
  store volatile i32 %D, ptr %P
  ; Reuse low bits of extended register, don't extend live range of SUM.
  ; CHECK-DAG: stw [[SUM]]
  %R = add i32 %D, %D
  ret i32 %R
}
