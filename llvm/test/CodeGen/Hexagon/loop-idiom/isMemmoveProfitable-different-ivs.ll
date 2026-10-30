; RUN: opt -passes='loop(loop-idiom),dce,loop(loop-deletion)' -S -mtriple=hexagon-unknown-elf < %s 2>&1 | FileCheck %s
;
; Test to check for profitable memmove replacement for different ivs (8 byte aligned) in Hexagon
;
; CHECK: @llvm.memmove

define void @test_different_ivs(ptr %a, i32 %n) {
entry:
  %guard = icmp sgt i32 %n, 0
  br i1 %guard, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  ; dst index starts at 0 (lower address), src index starts at 8 (higher).
  ; Both step +1 each iteration, keeping a constant 8-byte separation.
  %i = phi i32 [ 0, %loop.ph ], [ %i.next, %loop ]
  %j = phi i32 [ 8, %loop.ph ], [ %j.next, %loop ]

  %dst.ptr = getelementptr inbounds i8, ptr %a, i32 %i
  %src.ptr = getelementptr inbounds i8, ptr %a, i32 %j

  %val = load i8, ptr %src.ptr, align 1
  store i8 %val, ptr %dst.ptr, align 1

  %i.next = add nuw nsw i32 %i, 1
  %j.next = add nuw nsw i32 %j, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
