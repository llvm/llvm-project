; RUN: opt -disable-output -passes="print<scalar-evolution>" < %s 2>&1 | FileCheck %s

; Check that extending a nowrap add of a loop-invariant and an AddRec
; distributes the extension before the invariant is folded into the AddRec.
define void @foo(ptr %p, i1 %c, i32 %i0) {
; CHECK-LABEL: 'foo'
; CHECK-NEXT:  Classifying expressions for: @foo
; CHECK-NEXT:    %iv = phi i32 [ %i0, %entry ], [ %next, %loop ]
; CHECK-NEXT:    --> {%i0,+,1}<%loop> U: full-set S: full-set Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %offset0 = mul i32 %iv, 2
; CHECK-NEXT:    --> {(2 * %i0),+,2}<%loop> U: [0,-1) S: [-2147483648,2147483647) Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %offset1 = add nsw i32 %offset0, 1
; CHECK-NEXT:    --> {(1 + (2 * %i0))<nuw><nsw>,+,2}<%loop> U: full-set S: full-set Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %sindex0 = sext i32 %offset0 to i64
; CHECK-NEXT:    --> (sext i32 {(2 * %i0),+,2}<%loop> to i64) U: [0,-1) S: [-2147483648,2147483647) Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %sindex1 = sext i32 %offset1 to i64
; CHECK-NEXT:    --> (1 + (sext i32 {(2 * %i0),+,2}<%loop> to i64))<nuw><nsw> U: [1,0) S: [-2147483647,2147483648) Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %offset2 = add nuw i32 %offset0, 1
; CHECK-NEXT:    --> {(1 + (2 * %i0))<nuw><nsw>,+,2}<%loop> U: full-set S: full-set Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %uindex0 = zext i32 %offset0 to i64
; CHECK-NEXT:    --> (zext i32 {(2 * %i0),+,2}<%loop> to i64) U: [0,4294967295) S: [0,4294967296) Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %uindex1 = zext i32 %offset2 to i64
; CHECK-NEXT:    --> (1 + (zext i32 {(2 * %i0),+,2}<%loop> to i64))<nuw><nsw> U: [1,4294967296) S: [1,4294967297) Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %next = add i32 %iv, 1
; CHECK-NEXT:    --> {(1 + %i0),+,1}<%loop> U: full-set S: full-set Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:  Determining loop execution counts for: @foo
; CHECK-NEXT:  Loop %loop: Unpredictable backedge-taken count.
; CHECK-NEXT:  Loop %loop: Unpredictable constant max backedge-taken count.
; CHECK-NEXT:  Loop %loop: Unpredictable symbolic max backedge-taken count.
;
entry:
  br label %loop

loop:
  %iv = phi i32 [ %i0, %entry ], [ %next, %loop ]
  %offset0 = mul i32 %iv, 2
  %offset1 = add nsw i32 %offset0, 1
  %sindex0 = sext i32 %offset0 to i64
  %sindex1 = sext i32 %offset1 to i64
  %offset2 = add nuw i32 %offset0, 1
  %uindex0 = zext i32 %offset0 to i64
  %uindex1 = zext i32 %offset2 to i64
  %next = add i32 %iv, 1
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
