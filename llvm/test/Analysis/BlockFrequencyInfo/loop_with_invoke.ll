; RUN: opt < %s -analyze -block-freq -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Printing analysis {{.*}} for function 'loop_with_invoke':
; CHECK-NEXT: block-frequency-info: loop_with_invoke
define void @loop_with_invoke(i32 %n) personality i8 0 {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %loop

; CHECK-NEXT: loop: float = 9905.6
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %invoke.cont ]
  invoke void @foo() to label %invoke.cont unwind label %lpad

; CHECK-NEXT: invoke.cont: float = 9905.6
invoke.cont:
  %i.next = add i32 %i, 1
  %cont = icmp ult i32 %i.next, %n
  br i1 %cont, label %loop, label %exit, !prof !0

; CHECK-NEXT: lpad: float = 0.0094467
lpad:
  %ll = landingpad { i8*, i32 }
          cleanup
  br label %exit

; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret void
}

declare void @foo()

!0 = !{!"branch_weights", i32 9999, i32 1}
