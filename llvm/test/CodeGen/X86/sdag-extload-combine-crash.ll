; This test checks that after SelectionDAG runs, DAGCombiner can combine a
; load and a zext instruction without crashing in tryToFoldExtOfLoad(),
; even when the folded load result is still used along the truncate
; replacement path.

; RUN: llc -O3 -mtriple=x86_64-unknown-linux-gnu -filetype=null < %s

@ak = external global i16
@s = external global i16

define i32 @main() {
entry:
  %0 = load i16, ptr @ak, align 2
  %1 = load volatile i16, ptr @s, align 2
  %2 = load i16, ptr @ak, align 2
  %3 = xor i16 %2, -1
  %conv4211404 = zext i16 %3 to i64
  %xor422 = xor i64 1, %conv4211404
  %conv3591399 = zext i16 %0 to i64
  %or424 = or i64 %xor422, %conv3591399
  %conv333 = sext i16 %0 to i32
  %conv434 = zext i16 %2 to i32
  %4 = trunc i64 %or424 to i32
  %or436 = or i32 %conv434, %4
  %conv453 = xor i32 %conv333, %or436
  ret i32 %conv453
}
