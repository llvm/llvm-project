; RUN: opt < %s -loop-rotate -S | FileCheck %s

; Test alloca in -loop-rotate.

; We expect a different value for %ptr each iteration (according to the
; definition of alloca). I.e. each @use must be paired with an alloca.

; CHECK: call void @use(i8* %
; CHECK: %ptr = alloca i8

@e = global i16 10

declare void @use(i8*)

define void @test() {
entry:
  %end = load i16* @e
  br label %loop

loop:
  %n.phi = phi i16 [ %n, %loop.fin ], [ 0, %entry ]
  %ptr = alloca i8
  %cond = icmp eq i16 %n.phi, %end
  br i1 %cond, label %exit, label %loop.fin

loop.fin:
  %n = add i16 %n.phi, 1
  call void @use(i8* %ptr)
  br label %loop

exit:
  ret void
}
