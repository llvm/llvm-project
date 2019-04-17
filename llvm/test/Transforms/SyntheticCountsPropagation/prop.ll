; RUN: opt -passes=synthetic-counts-propagation -S < %s | FileCheck %s

; CHECK-LABEL: define void @level1a(i32 %n)
; CHECK: !prof ![[COUNT1:[0-9]+]]
define void @level1a(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 10
  br i1 %cmp, label %exit, label %loop
loop:
  %i = phi i32 [%n, %entry], [%i1, %loop]
  call void @level2a(i32 %n)
  %i1 = sub i32 %i, 1
  %cmp2 = icmp eq i32 %i1, 0
  br i1 %cmp2, label %exit, label %loop, !prof !1
exit:
  ret void
}

; CHECK-LABEL: define void @level2a(i32 %n)
; CHECK: !prof ![[COUNT2:[0-9]+]]
define void @level2a(i32 %n) {
  call void @level2b(i32 %n)
  ret void
}

; CHECK-LABEL: define void @level2b(i32 %n)
; CHECK: !prof ![[COUNT2]]
define void @level2b(i32 %n) {
entry:
  call void @level2a(i32 %n)
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %then, label %else, !prof !2
then:
  call void @level3a(i32 %n)
  br label %else
else:
  ret void
}

; CHECK-LABEL: define internal void @level3a(i32 %n)
; CHECK: !prof ![[COUNT3:[0-9]+]]
define internal void @level3a(i32 %n) {
  ret void
}

!1 = !{!"branch_weights", i32 1, i32 99}
!2 = !{!"branch_weights", i32 1, i32 1}
; CHECK: ![[COUNT1]] = !{!"synthetic_function_entry_count", i64 10}
; CHECK: ![[COUNT2]] = !{!"synthetic_function_entry_count", i64 520}
; CHECK: ![[COUNT3]] = !{!"synthetic_function_entry_count", i64 260}
