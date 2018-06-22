; RUN: opt < %s -cg-profile -S | FileCheck %s

declare void @b()

define void @a() !prof !1 {
  call void @b()
  ret void
}

define void @freq(i1 %cond) !prof !1 {
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

; CHECK: !llvm.module.flags = !{![[cgprof:[0-9]+]]}
; CHECK: ![[cgprof]] = !{i32 5, !"CG Profile", ![[prof:[0-9]+]]}
; CHECK: ![[prof]] = !{![[e0:[0-9]+]], ![[e1:[0-9]+]], ![[e2:[0-9]+]]}
; CHECK: ![[e0]] = !{void ()* @a, void ()* @b, i64 32}
; CHECK: ![[e1]] = !{void (i1)* @freq, void ()* @a, i64 11}
; CHECK: ![[e2]] = !{void (i1)* @freq, void ()* @b, i64 20}
