; RUN: opt < %s -partial-inliner -S | FileCheck %s
; RUN: opt < %s -passes=partial-inliner -S | FileCheck %s
; RUN: opt < %s -partial-inliner -skip-partial-inlining-cost-analysis -max-num-inline-blocks=2 -S | FileCheck --check-prefix=LIMIT %s
; RUN: opt < %s -passes=partial-inliner -skip-partial-inlining-cost-analysis -max-num-inline-blocks=2 -S | FileCheck  --check-prefix=LIMIT %s

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb5

bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 (...) @channels() #2
  %tmp3 = icmp slt i32 %tmp2, %arg
  br i1 %tmp3, label %bb4, label %bb5

bb4:                                              ; preds = %bb1
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  br label %bb5

bb5:                                              ; preds = %bb4, %bb1, %bb
  %tmp6 = phi i32 [ 0, %bb4 ], [ 1, %bb1 ], [ 1, %bb ]
  ret i32 %tmp6
}

declare i32 @channels(...) local_unnamed_addr #1

declare void @foo(...) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define i32 @dummy_caller(i32 %arg) local_unnamed_addr #0 {
bb:
; CHECK-LABEL: @dummy_caller
; CHECK: br i1
; CHECK: br i1
; CHECK: call void @bar.1_
; LIMIT-LABEL: @dummy_caller
; LIMIT: br i1
; LIMIT-NOT: br
; LIMIT: call void @bar.1_
  %tmp = tail call i32 @bar(i32 %arg)
  ret i32 %tmp
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { nounwind }

