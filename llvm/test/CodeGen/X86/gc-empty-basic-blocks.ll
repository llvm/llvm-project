;; This test verifies that -gc-empty-basic-blocks removes empty blocks.
; RUN: llc < %s -mtriple=x86_64 -O0 -gc-empty-basic-blocks | FileCheck -check-prefix=CHECK %s 
; RUN: llc < %s -mtriple=x86_64 -stats -O0 -gc-empty-basic-blocks 2>&1 | FileCheck -check-prefix=STAT %s

; STAT:        1 gc-empty-basic-blocks     - Number of empty blocks removed

define void @foo(i1 zeroext %0) nounwind {
  br i1 %0, label %2, label %empty_block

; CHECK:        .text
; CHECK-LABEL: foo:
; CHECK:         jne .LBB0_1
; CHECK-NEXT:    jmp .LBB0_3

2:                                               ; preds = %1
  %3 = call i32 @bar()
  br label %4

; CHECK-LABEL: .LBB0_1:
; CHECK:	 jmp .LBB0_3

empty_block:                                     ; preds = %1
  unreachable

; CHECK-NOT: %empty_block
; CHECK-NOT: .LBB0_2

4:                                               ; preds = %2, %empty_block
  ret void

; CHECK-LABEL: .LBB0_3:
; CHECK:         retq

}

declare i32 @bar()
