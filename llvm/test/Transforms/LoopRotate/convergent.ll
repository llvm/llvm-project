; RUN: opt -S -loop-rotate -verify-memoryssa < %s | FileCheck %s

@e = global i32 10

declare void @f1(i32) convergent
declare void @f2(i32)

; The call to f1 in the loop header shouldn't be duplicated (meaning, loop
; rotation shouldn't occur), because f1 is convergent.

; CHECK: call void @f1
; CHECK-NOT: call void @f1

define void @test(i32 %x) {
entry:
  br label %loop

loop:
  %n.phi = phi i32 [ %n, %loop.fin ], [ 0, %entry ]
  call void @f1(i32 %n.phi)
  %cond = icmp eq i32 %n.phi, %x
  br i1 %cond, label %exit, label %loop.fin

loop.fin:
  %n = add i32 %n.phi, 1
  call void @f2(i32 %n)
  br label %loop

exit:
  ret void
}
