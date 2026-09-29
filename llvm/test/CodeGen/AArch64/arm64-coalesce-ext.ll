; RUN: llc -mtriple=arm64-apple-darwin < %s | FileCheck %s
; Check that the peephole optimizer knows about sext and zext instructions.
; CHECK: test1sext
define i32 @test1sext(i64 %A, i64 %B, ptr %P, ptr %P2) nounwind {
  %C = add i64 %A, %B
  ; CHECK: add x[[SUM:[0-9]+]], x0, x1
  %D = trunc i64 %C to i32
  %E = shl i64 %C, 32
  %F = ashr i64 %E, 32
  ; CHECK: sxtw x[[EXT:[0-9]+]], w[[SUM]]
  store volatile i64 %F, ptr %P2
  ; CHECK: str x[[EXT]]
  store volatile i32 %D, ptr %P
  ; Reuse low bits of extended register, don't extend live range of SUM.
  ; CHECK: str w[[SUM]]
  ret i32 %D
}
