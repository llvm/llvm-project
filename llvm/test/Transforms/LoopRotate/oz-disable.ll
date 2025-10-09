; REQUIRES: asserts
; RUN: opt < %s -S -Os -debug -debug-only=loop-rotate 2>&1 | FileCheck %s -check-prefix=OS
; RUN: opt < %s -S -Oz -debug -debug-only=loop-rotate 2>&1 | FileCheck %s -check-prefix=OZ
; RUN: opt < %s -S -passes='default<Os>' -debug -debug-only=loop-rotate 2>&1 | FileCheck %s -check-prefix=OS
; RUN: opt < %s -S -passes='default<Oz>' -debug -debug-only=loop-rotate 2>&1 | FileCheck %s -check-prefix=OZ

;; Make sure -allow-loop-header-duplication overrides the default behavior at Oz
; RUN: opt < %s -S -passes='default<Oz>' -enable-loop-header-duplication -debug -debug-only=loop-rotate 2>&1 | FileCheck %s -check-prefix=OS

; Loop should be rotated for -Os but not for -Oz.
; OS: rotating Loop at depth 1
; OZ-NOT: rotating Loop at depth 1

@e = global i32 10

declare void @use(i32)

define void @test() {
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
