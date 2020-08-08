; RUN: opt -S -indvars -loop-unswitch -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -S -indvars -loop-unswitch -enable-new-pm=0 -enable-mssa-loop-dependency=true -verify-memoryssa < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @test_01() {

; Make sure we don't fail by SCEV's assertion due to incorrect invalidation.
; CHECK-LABEL: @test_01

entry:
  br label %loop

loop:                           ; preds = %backedge, %entry
  %p_50.addr.0 = phi i16 [ undef, %entry ], [ %add2699, %backedge ]
  %idxprom2690 = sext i16 %p_50.addr.0 to i32
  %arrayidx2691 = getelementptr inbounds [5 x i32], [5 x i32]* undef, i32 0, i32 %idxprom2690
  %0 = load i32, i32* %arrayidx2691, align 1
  %tobool2692 = icmp ne i32 %0, 0
  br label %inner_loop

inner_loop:                                     ; preds = %inner_backedge, %loop
  br i1 %tobool2692, label %backedge, label %inner_backedge

inner_backedge:                                       ; preds = %inner_loop
  br label %inner_loop

backedge:                                      ; preds = %inner_loop
  %add2699 = add nsw i16 %p_50.addr.0, 1
  br i1 false, label %loop, label %exit

exit:               ; preds = %backedge
  unreachable
}
