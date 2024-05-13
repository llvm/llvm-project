; RUN: opt < %s -passes=partial-inliner -skip-partial-inlining-cost-analysis -S | FileCheck %s

; This test checks to make sure that the CodeExtractor
;  properly sets the entry count for the function that is
;  extracted based on the root block being extracted and also
;  takes into consideration if the block has edges coming from
;  a block that is also being extracted.

define i32 @inlinedFunc(i1 %cond) !prof !1 {
entry:
  br i1 %cond, label %if.then, label %return, !prof !2
if.then:
  br i1 %cond, label %if.then, label %return, !prof !3
return:             ; preds = %entry
  ret i32 0
}


define internal i32 @dummyCaller(i1 %cond) !prof !1 {
entry:
  %val = call i32 @inlinedFunc(i1 %cond)
  ret i32 %val
}

; CHECK: @inlinedFunc.1.if.then(i1 %cond) !prof [[COUNT1:![0-9]+]]


!llvm.module.flags = !{!0}
; CHECK: [[COUNT1]] = !{!"function_entry_count", i64 250}
!0 = !{i32 1, !"MaxFunctionCount", i32 1000}
!1 = !{!"function_entry_count", i64 1000}
!2 = !{!"branch_weights", i32 250, i32 750}
!3 = !{!"branch_weights", i32 125, i32 125}
