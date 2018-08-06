; RUN: opt < %s -passes cg-profile -S | FileCheck %s

declare void @b()

define void @a() !prof !1 {
  call void @b()
  ret void
}

@foo = common global i32 ()* null, align 8
declare i32 @func1()
declare i32 @func2()
declare i32 @func3()
declare i32 @func4()

define void @freq(i1 %cond) !prof !1 {
  %tmp = load i32 ()*, i32 ()** @foo, align 8
  call i32 %tmp(), !prof !3
  br i1 %cond, label %A, label %B, !prof !2
A:
  call void @a();
  ret void
B:
  call void @b();
  ret void
}

!1 = !{!"function_entry_count", i64 32}
!2 = !{!"branch_weights", i32 5, i32 10}
!3 = !{!"VP", i32 0, i64 1600, i64 7651369219802541373, i64 1030, i64 -4377547752858689819, i64 410, i64 -6929281286627296573, i64 150, i64 -2545542355363006406, i64 10}

; CHECK: !llvm.module.flags = !{![[cgprof:[0-9]+]]}
; CHECK: ![[cgprof]] = !{i32 5, !"CG Profile", ![[prof:[0-9]+]]}
; CHECK: ![[prof]] = !{![[e0:[0-9]+]], ![[e1:[0-9]+]], ![[e2:[0-9]+]], ![[e3:[0-9]+]], ![[e4:[0-9]+]], ![[e5:[0-9]+]], ![[e6:[0-9]+]]}
; CHECK: ![[e0]] = !{void ()* @a, void ()* @b, i64 32}
; CHECK: ![[e1]] = !{void (i1)* @freq, i32 ()* @func4, i64 1030}
; CHECK: ![[e2]] = !{void (i1)* @freq, i32 ()* @func2, i64 410}
; CHECK: ![[e3]] = !{void (i1)* @freq, i32 ()* @func3, i64 150}
; CHECK: ![[e4]] = !{void (i1)* @freq, i32 ()* @func1, i64 10}
; CHECK: ![[e5]] = !{void (i1)* @freq, void ()* @a, i64 11}
; CHECK: ![[e6]] = !{void (i1)* @freq, void ()* @b, i64 20}
