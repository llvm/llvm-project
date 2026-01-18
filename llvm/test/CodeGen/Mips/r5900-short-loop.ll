; RUN: llc -mtriple=mips64el -mcpu=r5900 < %s | FileCheck %s -check-prefix=FIX
;
; Test R5900 short loop erratum fix.
; The R5900 has a hardware bug where short loops (6 instructions or fewer)
; with a branch may exit after 1-2 iterations instead of the expected count.
; The fix ensures the delay slot contains a NOP for such short backward branches.
; The fix is always enabled for R5900 - there is no way to disable it.

; Short loop test with store - delay slot NOT filled due to short loop fix
; bnez followed by nop (short loop workaround)

; FIX-LABEL: test_short_loop_store:
; FIX:       .LBB0_1:
; FIX:       bnez
; FIX-NEXT:  nop
define void @test_short_loop_store(ptr %arr, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %ptr = getelementptr i32, ptr %arr, i32 %i
  store i32 %i, ptr %ptr
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Long loop - should NOT be affected by the fix (too many instructions)
; Delay slot can be filled since loop is longer than 6 instructions
; The scheduler may choose different instructions for the delay slot
; FIX-LABEL: test_long_loop:
; FIX:       .LBB1_1:
; FIX:       bnez ${{[0-9]+}}, .LBB1_1
; Delay slot should be filled with actual instruction, not nop
; FIX-NEXT:  {{sw|daddiu|addu}}
define i32 @test_long_loop(ptr %arr, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %add, %loop ]
  %ptr = getelementptr i32, ptr %arr, i32 %i
  %val = load i32, ptr %ptr
  %mul = mul i32 %val, %i
  %add = add i32 %sum, %mul
  %ptr2 = getelementptr i32, ptr %arr, i32 %add
  store i32 %add, ptr %ptr2
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret i32 %add
}
