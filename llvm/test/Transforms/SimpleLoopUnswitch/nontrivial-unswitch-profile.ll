; RUN: split-file %s %t
; RUN: cat %t/main.ll %t/probable-or.prof > %t/probable-or.ll
; RUN: cat %t/main.ll %t/probable-and.prof > %t/probable-and.ll
; RUN: opt -passes='loop(simple-loop-unswitch<nontrivial>)' -S %t/probable-or.ll -o -| FileCheck %t/probable-or.prof
; RUN: opt -passes='loop(simple-loop-unswitch<nontrivial>)' -S %t/probable-and.ll -o -| FileCheck %t/probable-and.prof

;--- main.ll
declare i32 @a()
declare i32 @b()

define i32 @or(ptr %ptr, i1 %cond) !prof !0 {
entry:
  br label %loop_begin

loop_begin:
  %v1 = load i1, ptr %ptr
  %cond_or = or i1 %v1, %cond
  br i1 %cond_or, label %loop_a, label %loop_b, !prof !1

loop_a:
  call i32 @a()
  br label %latch

loop_b:
  call i32 @b()
  br label %latch

latch:
  %v2 = load i1, ptr %ptr
  br i1 %v2, label %loop_begin, label %loop_exit, !prof !2

loop_exit:
  ret i32 0
}

define i32 @and(ptr %ptr, i1 %cond) !prof !0 {
entry:
  br label %loop_begin

loop_begin:
  %v1 = load i1, ptr %ptr
  %cond_and = and i1 %v1, %cond
  br i1 %cond_and, label %loop_a, label %loop_b, !prof !1

loop_a:
  call i32 @a()
  br label %latch

loop_b:
  call i32 @b()
  br label %latch

latch:
  %v2 = load i1, ptr %ptr
  br i1 %v2, label %loop_begin, label %loop_exit, !prof !2

loop_exit:
  ret i32 0
}

;--- probable-or.prof
!0 = !{!"function_entry_count", i32 10}
!1 = !{!"branch_weights", i32 1, i32 1000}
!2 = !{!"branch_weights", i32 5, i32 7}
; CHECK-LABEL: @or
; CHECK-LABEL: entry:
; CHECK-NEXT:   %cond.fr = freeze i1 %cond
; CHECK-NEXT:   br i1 %cond.fr, label %entry.split.us, label %entry.split, !prof !1
; CHECK-LABEL: @and
; CHECK-LABEL: entry:
; CHECK-NEXT:   %cond.fr = freeze i1 %cond
; CHECK-NEXT:   br i1 %cond.fr, label %entry.split, label %entry.split.us, !prof !3
; CHECK: !1 = !{!"branch_weights", i32 1, i32 1000}
; CHECK: !3 = !{!"unknown", !"simple-loop-unswitch"}

;--- probable-and.prof
!0 = !{!"function_entry_count", i32 10}
!1 = !{!"branch_weights", i32 1000, i32 1}
!2 = !{!"branch_weights", i32 5, i32 7}
; CHECK-LABEL: @or
; CHECK-LABEL: entry:
; CHECK-NEXT:   %cond.fr = freeze i1 %cond
; CHECK-NEXT:   br i1 %cond.fr, label %entry.split.us, label %entry.split, !prof !1
; CHECK-LABEL: @and
; CHECK-LABEL: entry:
; CHECK-NEXT:   %cond.fr = freeze i1 %cond
; CHECK-NEXT:   br i1 %cond.fr, label %entry.split, label %entry.split.us, !prof !3
; CHECK: !1 = !{!"unknown", !"simple-loop-unswitch"}
; CHECK: !3 = !{!"branch_weights", i32 1000, i32 1}
