; RUN: opt < %s -loop-reroll -S | FileCheck %s

; Check whether rerolling is rejected if values of the base and root
; instruction are used outside the loop block.

; Only the base/root instructions except a loop increment instruction
define void @test1() {
entry:
  br label %loop1

loop1:
;CHECK-LABEL: loop1:
;CHECK-NEXT:   %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop1 ]
;CHECK-NEXT:   %indvar.1 = add nsw i64 %indvar, 1

  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop1 ]
  %indvar.1 = add nsw i64 %indvar, 1
  %indvar.next = add nsw i64 %indvar, 2
  %cmp = icmp slt i64 %indvar.next, 200
  br i1 %cmp, label %loop1, label %exit

exit:
  %var1 = phi i64 [ %indvar.1, %loop1 ]
  %var2 = phi i64 [ %indvar, %loop1 ]
  ret void
}

; Both the base/root instructions and reduction instructions
define void @test2() {
entry:
  br label %loop2

loop2:
;CHECK-LABEL: loop2:
;CHECK-NEXT:   %indvar = phi i32  [ 0, %entry ], [ %indvar.next, %loop2 ]
;CHECK-NEXT:   %redvar = phi i32 [ 0, %entry ], [ %add.2, %loop2 ]
;CHECK-NEXT:   %indvar.1 = add nuw nsw i32 %indvar, 1
;CHECK-NEXT:   %indvar.2 = add nuw nsw i32 %indvar, 2

  %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %loop2 ]
  %redvar = phi i32 [ 0, %entry ], [ %add.2, %loop2 ]
  %indvar.1 = add nuw nsw i32 %indvar, 1
  %indvar.2 = add nuw nsw i32 %indvar, 2
  %mul.0 = mul nsw i32 %indvar, %indvar
  %mul.1 = mul nsw i32 %indvar.1, %indvar.1
  %mul.2 = mul nsw i32 %indvar.2, %indvar.2
  %add.0 = add nsw i32 %redvar, %mul.0
  %add.1 = add nsw i32 %add.0, %mul.1
  %add.2 = add nsw i32 %add.1, %mul.2
  %indvar.next = add nuw nsw i32 %indvar, 3
  %cmp = icmp slt i32 %indvar.next, 300
  br i1 %cmp, label %loop2, label %exit

exit:
  %a = phi i32 [ %indvar, %loop2 ]
  %b = phi i32 [ %indvar.1, %loop2 ]
  %c = phi i32 [ %indvar.2, %loop2 ]
  %x = phi i32 [ %add.2, %loop2 ]
  ret void
}
