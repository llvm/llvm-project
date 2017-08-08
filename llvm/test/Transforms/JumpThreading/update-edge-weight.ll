; RUN: opt -S -jump-threading %s | FileCheck %s

; Test if edge weights are properly updated after jump threading.

; CHECK: !2 = !{!"branch_weights", i32 1629125526, i32 518358122}

define void @foo(i32 %n) !prof !0 {
entry:
  %cmp = icmp sgt i32 %n, 10
  br i1 %cmp, label %if.then.1, label %if.else.1, !prof !1

if.then.1:
  tail call void @a()
  br label %if.cond

if.else.1:
  tail call void @b()
  br label %if.cond

if.cond:
  %cmp1 = icmp sgt i32 %n, 5
  br i1 %cmp1, label %if.then.2, label %if.else.2, !prof !2

if.then.2:
  tail call void @c()
  br label %if.end

if.else.2:
  tail call void @d()
  br label %if.end

if.end:
  ret void
}

declare void @a()
declare void @b()
declare void @c()
declare void @d()

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 10, i32 5}
!2 = !{!"branch_weights", i32 10, i32 1}
