; RUN: llc -mtriple=aarch64-apple-darwin -aarch64-bcc-offset-bits=4 -align-all-nofallthru-blocks=4 < %s | FileCheck %s

; Long branch is assumed because the block has a higher alignment
; requirement than the function.

; CHECK-LABEL: invert_bcc_block_align_higher_func:
; CHECK: b.eq [[JUMP_BB1:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: b [[JUMP_BB2:LBB[0-9]+_[0-9]+]]

; CHECK: [[JUMP_BB1]]:
; CHECK: ret
; CHECK: .p2align 4

; CHECK: [[JUMP_BB2]]:
; CHECK: ret
define i32 @invert_bcc_block_align_higher_func(i32 %x, i32 %y) align 4 #0 {
  %1 = icmp eq i32 %x, %y
  br i1 %1, label %bb1, label %bb2

bb2:
  store volatile i32 9, i32* undef
  ret i32 1

bb1:
  store volatile i32 42, i32* undef
  ret i32 0
}

attributes #0 = { nounwind }
