; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

declare void @g(i32 %x)

; CHECK-LABEL: Printing analysis {{.*}} for function 'branch_weight_0':
; CHECK-NEXT: block-frequency-info: branch_weight_0
define void @branch_weight_0(i32 %a) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %for.body

; Check that we get 1 and a huge frequency instead of 0,3.
; CHECK-NEXT: for.body: float = 2147483647.8,
for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void @g(i32 %i)
  %inc = add i32 %i, 1
  %cmp = icmp ugt i32 %inc, %a
  br i1 %cmp, label %for.end, label %for.body, !prof !0

; CHECK-NEXT: for.end: float = 1.0, int = [[ENTRY]]
for.end:
  ret void
}

!0 = !{!"branch_weights", i32 0, i32 3}

; CHECK-LABEL: Printing analysis {{.*}} for function 'infinite_loop'
; CHECK-NEXT: block-frequency-info: infinite_loop
define void @infinite_loop(i1 %x) {
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br i1 %x, label %for.body, label %for.end, !prof !1

; Check that the infinite loop is arbitrarily scaled to max out at 4096,
; giving 2048 here.
; CHECK-NEXT: for.body: float = 2048.0,
for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void @g(i32 %i)
  %inc = add i32 %i, 1
  br label %for.body

; Check that the exit weight is half of entry, since half is lost in the
; infinite loop above.
; CHECK-NEXT: for.end: float = 0.5,
for.end:
  ret void
}

!1 = !{!"branch_weights", i32 1, i32 1}
