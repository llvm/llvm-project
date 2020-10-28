; RUN: opt < %s -analyze -block-freq -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Printing analysis {{.*}} for function 'loop_with_branch':
; CHECK-NEXT: block-frequency-info: loop_with_branch
define void @loop_with_branch(i32 %a) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  %skip_loop = call i1 @foo0(i32 %a)
  br i1 %skip_loop, label %skip, label %header, !prof !0

; CHECK-NEXT: skip: float = 0.25,
skip:
  br label %exit

; CHECK-NEXT: header: float = 4.5,
header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %back ]
  %i.next = add i32 %i, 1
  %choose = call i2 @foo1(i32 %i)
  switch i2 %choose, label %exit [ i2 0, label %left
                                   i2 1, label %right ], !prof !1

; CHECK-NEXT: left: float = 1.5,
left:
  br label %back

; CHECK-NEXT: right: float = 2.25,
right:
  br label %back

; CHECK-NEXT: back: float = 3.75,
back:
  br label %header

; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret void
}

declare i1 @foo0(i32)
declare i2 @foo1(i32)

!0 = !{!"branch_weights", i32 1, i32 3}
!1 = !{!"branch_weights", i32 1, i32 2, i32 3}
