; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; PR21622: Check for a crasher when the sum of exits to the same successor of a
; loop overflows.

; CHECK-LABEL: Printing analysis {{.*}} for function 'extremely_likely_loop_successor':
; CHECK-NEXT: block-frequency-info: extremely_likely_loop_successor
define void @extremely_likely_loop_successor() {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %loop

; CHECK-NEXT: loop: float = 1.0,
loop:
  %exit.1.cond = call i1 @foo()
  br i1 %exit.1.cond, label %exit, label %loop.2, !prof !0

; CHECK-NEXT: loop.2: float = 0.0000000
loop.2:
  %exit.2.cond = call i1 @foo()
  br i1 %exit.2.cond, label %exit, label %loop.3, !prof !0

; CHECK-NEXT: loop.3: float = 0.0000000
loop.3:
  %exit.3.cond = call i1 @foo()
  br i1 %exit.3.cond, label %exit, label %loop.4, !prof !0

; CHECK-NEXT: loop.4: float = 0.0,
loop.4:
  %exit.4.cond = call i1 @foo()
  br i1 %exit.4.cond, label %exit, label %loop, !prof !0

; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret void
}

declare i1 @foo()

!0 = !{!"branch_weights", i32 4294967295, i32 1}
