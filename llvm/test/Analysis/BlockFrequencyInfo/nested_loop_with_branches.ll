; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Printing analysis {{.*}} for function 'nested_loop_with_branches'
; CHECK-NEXT: block-frequency-info: nested_loop_with_branches
define void @nested_loop_with_branches(i32 %a) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  %v0 = call i1 @foo0(i32 %a)
  br i1 %v0, label %exit, label %outer, !prof !0

; CHECK-NEXT: outer: float = 12.0,
outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %inner.end ], [ %i.next, %no_inner ]
  %i.next = add i32 %i, 1
  %do_inner = call i1 @foo1(i32 %i)
  br i1 %do_inner, label %no_inner, label %inner, !prof !0

; CHECK-NEXT: inner: float = 36.0,
inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner.end ]
  %side = call i1 @foo3(i32 %j)
  br i1 %side, label %left, label %right, !prof !0

; CHECK-NEXT: left: float = 9.0,
left:
  %v4 = call i1 @foo4(i32 %j)
  br label %inner.end

; CHECK-NEXT: right: float = 27.0,
right:
  %v5 = call i1 @foo5(i32 %j)
  br label %inner.end

; CHECK-NEXT: inner.end: float = 36.0,
inner.end:
  %stay_inner = phi i1 [ %v4, %left ], [ %v5, %right ]
  %j.next = add i32 %j, 1
  br i1 %stay_inner, label %inner, label %outer, !prof !1

; CHECK-NEXT: no_inner: float = 3.0,
no_inner:
  %continue = call i1 @foo6(i32 %i)
  br i1 %continue, label %outer, label %exit, !prof !1

; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret void
}

declare i1 @foo0(i32)
declare i1 @foo1(i32)
declare i1 @foo2(i32)
declare i1 @foo3(i32)
declare i1 @foo4(i32)
declare i1 @foo5(i32)
declare i1 @foo6(i32)

!0 = !{!"branch_weights", i32 1, i32 3}
!1 = !{!"branch_weights", i32 3, i32 1}
