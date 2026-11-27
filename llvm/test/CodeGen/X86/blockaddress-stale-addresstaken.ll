; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+retpoline | FileCheck %s
;
; verify that blocks are NOT marked as "Block address taken" when the
; BlockAddress constant has no users (was optimized away).
;
; With retpoline enabled, the indirectbr is replaced with direct comparisons
; against constant integer values. The BlockAddress constants in the global
; array become unused (constant-folded to integers), so the blocks should NOT
; be marked as address-taken.

@targets = internal constant [4 x ptr] [
  ptr blockaddress(@test_stale_addresstaken, %bb0),
  ptr blockaddress(@test_stale_addresstaken, %bb1),
  ptr blockaddress(@test_stale_addresstaken, %bb2),
  ptr blockaddress(@test_stale_addresstaken, %bb3)
]

define i32 @test_stale_addresstaken(i32 %idx) {
entry:
  %ptr = getelementptr [4 x ptr], ptr @targets, i32 0, i32 %idx
  %dest = load ptr, ptr %ptr
  indirectbr ptr %dest, [label %bb0, label %bb1, label %bb2, label %bb3]

; CHECK-LABEL: test_stale_addresstaken:
; CHECK-NOT: Block address taken
; CHECK-LABEL: .Lfunc_end0:

bb0:
  ret i32 0

bb1:
  ret i32 1

bb2:
  ret i32 2

bb3:
  ret i32 3
}
