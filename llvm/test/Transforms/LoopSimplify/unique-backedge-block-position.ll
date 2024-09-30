; RUN: opt < %s -passes='loop-simplify' -S | FileCheck %s

; llvm/llvm-project#15968: loop-simplify's insertUniqueBackedgeBlock inserted
; in the middle of blocks, instead of past the final backedge (as documented)

; CHECK: define void @test_function
define void @test_function() {
entry:
  br label %loop_header

; CHECK: loop_header:
loop_header:
  %i = phi i32 [ 0, %entry ], [ %next_value_1, %backedge_block_1 ], [ %next_value_2, %backedge_block_2 ]

  %condition = icmp slt i32 %i, 5
  br i1 %condition, label %backedge_block_1, label %backedge_block_2

; CHECK: backedge_block_1:
backedge_block_1:
  %next_value_1 = add i32 %i, 1
  br label %loop_header

; CHECK: backedge_block_2:
backedge_block_2:
  %next_value_2 = add i32 %i, 2
  br label %loop_header

; CHECK: loop_header.backedge:

loop_exit:
  ret void
}
