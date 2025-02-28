;; This test verifies that with -gc-empty-basic-blocks SHT_LLVM_BB_ADDR_MAP will not include entries for empty blocks.
; RUN: llc < %s -mtriple=x86_64 -O0 -basic-block-address-map -gc-empty-basic-blocks | FileCheck --check-prefix=CHECK %s

define void @foo(i1 zeroext %0) nounwind {
  br i1 %0, label %2, label %empty_block

2:                                               ; preds = %1
  %3 = call i32 @bar()
  br label %4

empty_block:                                     ; preds = %1
  unreachable

4:                                               ; preds = %2, %empty_block
  ret void
}

declare i32 @bar()

; CHECK: .section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
; CHECK: .byte	3                               # number of basic blocks
