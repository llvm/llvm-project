; RUN: llc -O0 -mtriple=aarch64-linux-gnu -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

; Test that the GlobalISel IRTranslator correctly marks blocks as address-taken
; based on whether the BlockAddress actually has users.

; CHECK-LABEL: name: test_indirectbr_blockaddress
; CHECK: G_BLOCK_ADDR blockaddress(@test_indirectbr_blockaddress, %ir-block.target)
; CHECK: G_BLOCK_ADDR blockaddress(@test_indirectbr_blockaddress, %ir-block.other)
; CHECK: G_BRINDIRECT
; CHECK: bb.{{[0-9]+}}.target (ir-block-address-taken %ir-block.target):
; CHECK: bb.{{[0-9]+}}.other (ir-block-address-taken %ir-block.other):
define i32 @test_indirectbr_blockaddress(i32 %idx) {
entry:
  %targets = alloca [2 x ptr], align 8
  %ptr0 = getelementptr [2 x ptr], ptr %targets, i64 0, i64 0
  store ptr blockaddress(@test_indirectbr_blockaddress, %target), ptr %ptr0, align 8
  %ptr1 = getelementptr [2 x ptr], ptr %targets, i64 0, i64 1
  store ptr blockaddress(@test_indirectbr_blockaddress, %other), ptr %ptr1, align 8
  %idx64 = zext i32 %idx to i64
  %selected = getelementptr [2 x ptr], ptr %targets, i64 0, i64 %idx64
  %dest = load ptr, ptr %selected, align 8
  indirectbr ptr %dest, [label %target, label %other]

target:
  ret i32 42

other:
  ret i32 -1
}

; normal conditional branch (no blockaddress).
; blocks should NOT be marked as address-taken.

; CHECK-LABEL: name: test_normal_branch
; CHECK: bb.{{[0-9]+}}.target:
; CHECK-NOT: ir-block-address-taken
; CHECK: bb.{{[0-9]+}}.other:
; CHECK-NOT: ir-block-address-taken
define i32 @test_normal_branch(i1 %cond) {
entry:
  br i1 %cond, label %target, label %other

target:
  ret i32 42

other:
  ret i32 -1
}
