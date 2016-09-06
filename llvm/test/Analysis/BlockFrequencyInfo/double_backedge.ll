; RUN: opt < %s -analyze -block-freq | FileCheck %s
; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

define void @double_backedge(i1 %x) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'double_backedge':
; CHECK-NEXT: block-frequency-info: double_backedge
entry:
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
  br label %loop

loop:
; CHECK-NEXT: loop: float = 10.0,
  br i1 %x, label %exit, label %loop.1, !prof !0

loop.1:
; CHECK-NEXT: loop.1: float = 9.0,
  br i1 %x, label %loop, label %loop.2, !prof !1

loop.2:
; CHECK-NEXT: loop.2: float = 5.0,
  br label %loop

exit:
; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
  ret void
}
!0 = !{!"branch_weights", i32 1, i32 9}
!1 = !{!"branch_weights", i32 4, i32 5}
