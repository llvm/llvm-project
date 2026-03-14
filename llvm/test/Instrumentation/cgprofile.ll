; RUN: opt < %s -passes='cg-profile<in-lto-post-link>' -S | FileCheck %s --check-prefixes=CHECK,LTO
; RUN: opt < %s -passes='cg-profile' -S | FileCheck %s --check-prefixes=CHECK,NOLTO --implicit-check-not="!{ptr @freq, ptr @func3.llvm.12345"

declare void @b()

define void @a() !prof !1 {
  call void @b()
  ret void
}

define void @func3.llvm.12345() !PGOFuncName !4 {
  ret void
}

@foo = common global ptr null, align 8
declare i32 @func1()
declare i32 @func2()

declare i32 @func4()
declare dllimport i32 @func5()
declare i32 @func6()

define void @freq(i1 %cond) !prof !1 {
  %tmp = load ptr, ptr @foo, align 8
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
!3 = !{!"VP", i32 0, i64 1600, i64 7651369219802541373, i64 1030, i64 -4377547752858689819, i64 410, i64 5415368997850289431, i64 150, i64 -2545542355363006406, i64 10, i64 3667884930908592509, i64 1, i64 15435711456043681792, i64 0}
!4 = !{!"cgprofile.ll;func3"}

; CHECK: !llvm.module.flags = !{![[cgprof:[0-9]+]]}
; CHECK: ![[cgprof]] = !{i32 5, !"CG Profile", ![[prof:[0-9]+]]}
; LTO: ![[prof]] = distinct !{![[e0:[0-9]+]], ![[e1:[0-9]+]], ![[e2:[0-9]+]], ![[e3:[0-9]+]], ![[e4:[0-9]+]], ![[e5:[0-9]+]], ![[e6:[0-9]+]]}
; NOLTO: ![[prof]] = distinct !{![[e0:[0-9]+]], ![[e1:[0-9]+]], ![[e2:[0-9]+]], ![[e4:[0-9]+]], ![[e5:[0-9]+]], ![[e6:[0-9]+]]}
; CHECK: ![[e0]] = !{ptr @a, ptr @b, i64 32}
; CHECK: ![[e1]] = !{ptr @freq, ptr @func4, i64 1030}
; CHECK: ![[e2]] = !{ptr @freq, ptr @func2, i64 410}
; LTO: ![[e3]] = !{ptr @freq, ptr @func3.llvm.12345, i64 150}
; CHECK: ![[e4]] = !{ptr @freq, ptr @func1, i64 10}
; CHECK: ![[e5]] = !{ptr @freq, ptr @a, i64 11}
; CHECK: ![[e6]] = !{ptr @freq, ptr @b, i64 21}
; CHECK-NOT: !{ptr @freq, ptr @func5, i64 1}
; CHECK-NOT: !{ptr @freq, ptr @func6, i64 0}
