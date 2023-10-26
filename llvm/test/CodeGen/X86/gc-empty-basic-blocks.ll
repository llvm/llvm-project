;; This test verifies that -gc-empty-basic-blocks removes regular empty blocks
;; but does not remove empty blocks which have their address taken.
; RUN: llc < %s -mtriple=x86_64 -O0 -gc-empty-basic-blocks | FileCheck %s

;; This function has a regular empty block.
define void @foo(i1 zeroext %0) nounwind {
  br i1 %0, label %2, label %empty_block

; CHECK:        .text
; CHECK-LABEL: foo:
; CHECK:         jne .LBB0_1
; CHECK-NEXT:    jmp .LBB0_3

2:                                               ; preds = %1
  %3 = call i32 @baz()
  br label %4

; CHECK-LABEL: .LBB0_1:
; CHECK:	 jmp .LBB0_3

empty_block:                                     ; preds = %1
  unreachable

; CHECK-NOT:     %empty_block
; CHECK-NOT:   .LBB0_2

4:                                               ; preds = %2, %empty_block
  ret void

; CHECK-LABEL: .LBB0_3:
; CHECK:         retq

}

;; This function has an empty block which has its address taken. Check that it
;; is not removed by -gc-empty-basic-blocks.
define void @bar(i1 zeroext %0) nounwind {
entry:
  %1 = select i1 %0, ptr blockaddress(@bar, %empty_block), ptr blockaddress(@bar, %bb2) ; <ptr> [#uses=1]
  indirectbr ptr %1, [label %empty_block, label %bb2]

; CHECK-LABEL: bar:

empty_block:                                                ; preds = %entry
  unreachable

; CHECK-LABEL: .LBB1_1: # %empty_block

bb2:                                                ; preds = %entry
  %2 = call i32 @baz()
  ret void
}

declare i32 @baz()
