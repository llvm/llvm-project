; REQUIRES: asserts
; RUN: opt < %s -S -passes=loop-rotate -debug -debug-only=loop-rotate 2>&1 | FileCheck %s

; Loop should not be rotated for functions with the minsize attribute.
; This is mostly useful for LTO which doesn't (yet) understand -Oz.
; CHECK: LoopRotation: NOT rotating - contains 2 instructions, which is more

@e = global i32 10

declare void @use(i32)

; Function attrs: minsize optsize
define void @test() #0 {
entry:
  %end = load i32, ptr @e
  br label %loop

loop:
  %n.phi = phi i32 [ %n, %loop.fin ], [ 0, %entry ]
  %cond = icmp eq i32 %n.phi, %end
  br i1 %cond, label %exit, label %loop.fin

loop.fin:
  %n = add i32 %n.phi, 1
  call void @use(i32 %n)
  br label %loop

exit:
  ret void
}

attributes #0 = { minsize optsize }
