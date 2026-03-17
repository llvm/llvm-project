; This test checks that profile data is preserved after GlobalMergeFunctions:
; - Merged function (.Tgm): preserves entry count and branch weights for blocks
; - Thunk: preserves its original entry count

; RUN: opt -mtriple=arm64-apple-darwin -S --passes=global-merge-func %s | FileCheck %s

; CHECK: @f1.Tgm(i32 %0, ptr %1){{.*}} !prof [[PROF1:![0-9]+]]
; CHECK: br i1 %cmp, label %if.then, label %if.end, !prof [[BRANCH:![0-9]+]]
; CHECK: @f1(i32 %a){{.*}} !prof [[PROF1]]

; CHECK: @f2.Tgm(i32 %0, ptr %1){{.*}} !prof [[PROF2:![0-9]+]]
; CHECK: br i1 %cmp, label %if.then, label %if.end, !prof [[BRANCH]]
; CHECK: @f2(i32 %a){{.*}} !prof [[PROF2]]

; CHECK-DAG: [[PROF1]] = !{!"function_entry_count", i64 1000}
; CHECK-DAG: [[PROF2]] = !{!"function_entry_count", i64 500}
; CHECK-DAG: [[BRANCH]] = !{!"branch_weights", i32 99, i32 1}

@g1 = external global i32, align 4
@g2 = external global i32, align 4

define i32 @f1(i32 %a) !prof !0 {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end, !prof !2

if.then:
  %0 = load volatile i32, ptr @g1, align 4
  %mul = mul nsw i32 %0, %a
  br label %if.end

if.end:
  %result = phi i32 [ %mul, %if.then ], [ %a, %entry ]
  ret i32 %result
}

define i32 @f2(i32 %a) !prof !1 {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end, !prof !2

if.then:
  %0 = load volatile i32, ptr @g2, align 4
  %mul = mul nsw i32 %0, %a
  br label %if.end

if.end:
  %result = phi i32 [ %mul, %if.then ], [ %a, %entry ]
  ret i32 %result
}

!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"function_entry_count", i64 500}
!2 = !{!"branch_weights", i32 99, i32 1}
