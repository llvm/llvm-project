; REQUIRES: asserts
; RUN: opt < %s -S -O2 -debug -debug-only=loop-rotate 2>&1 | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt < %s -S -passes='default<O2>' -debug -debug-only=loop-rotate 2>&1 | FileCheck %s --check-prefixes=CHECK,DEFAULT

;; Make sure -enable-loop-header-duplication-at-minsize overrides the default behavior at Oz
; RUN: opt < %s -S -passes='default<O2>' -enable-loop-header-duplication-at-minsize -debug -debug-only=loop-rotate 2>&1 | FileCheck %s -check-prefixes=CHECK,ALLOW

; optsize loop should always be rotated.
; CHECK: rotating Loop at depth 1
; minsize loop should only be rotated under the option.
; DEFAULT-NOT: rotating Loop at depth 1
; ALLOW: rotating Loop at depth 1

@e = global i32 10

declare void @use(i32)

define void @test_optsize() optsize {
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

define void @test_minsize() minsize {
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
