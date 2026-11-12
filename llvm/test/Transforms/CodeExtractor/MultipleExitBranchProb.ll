; RUN: opt < %s -passes=partial-inliner -max-num-inline-blocks=2 -skip-partial-inlining-cost-analysis -S | FileCheck %s

; This test checks to make sure that CodeExtractor updates
;  the exit branch probabilities for multiple exit blocks.

define i32 @inlinedFunc(i1 %cond) !prof !1 {
entry:
  br i1 %cond, label %if.then, label %return, !prof !2
if.then:
  br i1 %cond, label %return, label %return.2, !prof !3
return.2:
  ret i32 10
return:             ; preds = %entry
  ret i32 0
}


define internal i32 @dummyCaller(i1 %cond) !prof !1 {
entry:
%val = call i32 @inlinedFunc(i1 %cond)
ret i32 %val

; CHECK-LABEL: @dummyCaller
; CHECK: call
; CHECK-NEXT: br i1 {{.*}}return.i{{.*}}return.2{{.*}}!prof [[COUNT1:![0-9]+]]
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"MaxFunctionCount", i32 10000}
!1 = !{!"function_entry_count", i64 10000}
!2 = !{!"branch_weights", i32 5, i32 5}
!3 = !{!"branch_weights", i32 4, i32 1}

; CHECK: [[COUNT1]] = !{!"branch_weights", i32 858993459, i32 214748365}
