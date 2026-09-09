; RUN: opt < %s -passes=partial-inliner -skip-partial-inlining-cost-analysis -S | FileCheck %s
;
; CodeExtractor copies approxprofile onto the extracted function.

define i32 @inlinedFunc(i1 %cond) approxprofile !prof !1 {
entry:
  br i1 %cond, label %if.then, label %return, !prof !2
if.then:
  br i1 %cond, label %if.then, label %return, !prof !3
return:
  ret i32 0
}

define internal i32 @dummyCaller(i1 %cond) !prof !1 {
entry:
  %val = call i32 @inlinedFunc(i1 %cond)
  ret i32 %val
}

; CHECK: define {{.*}} @inlinedFunc.1.if.then({{.*}}) #[[A:[0-9]+]]
; CHECK: attributes #[[A]] = { {{.*}}approxprofile{{.*}} }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"MaxFunctionCount", i32 1000}
!1 = !{!"function_entry_count", i64 1000}
!2 = !{!"branch_weights", i32 250, i32 750}
!3 = !{!"branch_weights", i32 125, i32 125}
