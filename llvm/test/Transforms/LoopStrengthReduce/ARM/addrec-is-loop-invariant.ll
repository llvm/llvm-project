; RUN: llc -mtriple=armv8-eabi -verify-machineinstrs %s -o /dev/null

; This test ensures that Loop Strength Reduction will
; not create an Add Reccurence Expression if not all
; its operands are loop invariants.

define void @add_rec_expr() {
entry:
  br label %loop0

loop0:
  %c.0 = phi i32 [ 0, %entry ], [ %inc.0, %loop0 ]
  %inc.0 = add nuw i32 %c.0, 1
  br i1 undef, label %loop0, label %bb1

bb1:
  %mul.0 = mul i32 %c.0, %c.0
  %gelptr.0 = getelementptr inbounds i16, ptr undef, i32 %mul.0
  br label %loop1

loop1:
  %inc.1 = phi i32 [ %inc.2, %bb4 ], [ 0, %bb1 ]
  %mul.1 = mul i32 %inc.1, %c.0
  br label %bb3

bb3:
  %add.0 = add i32 undef, %mul.1
  %gelptr.1 = getelementptr inbounds i16, ptr %gelptr.0, i32 %add.0
  store i16 undef, ptr %gelptr.1, align 2
  br label %bb4

bb4:
  %inc.2 = add nuw i32 %inc.1, 1
  br label %loop1
}
