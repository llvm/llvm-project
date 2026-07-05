; RUN: opt < %s -passes=partial-inliner -skip-partial-inlining-cost-analysis -S | FileCheck %s
; RUN: opt < %s -passes=partial-inliner -max-num-inline-blocks=2 -S | FileCheck  --check-prefix=LIMIT %s

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb4, label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 (...) @channels() #1
  %tmp3 = icmp slt i32 %tmp2, %arg
  br i1 %tmp3, label %bb4, label %bb5

bb4:                                              ; preds = %bb1, %bb
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  br label %bb5

bb5:                                              ; preds = %bb4, %bb1
  %.0 = phi i32 [ 0, %bb4 ], [ 1, %bb1 ]
  ret i32 %.0
}

declare i32 @channels(...) local_unnamed_addr

declare void @foo(...) local_unnamed_addr

; Function Attrs: nounwind uwtable
define i32 @dummy_caller(i32 %arg) local_unnamed_addr #0 {
bb:
; CHECK-LABEL: @dummy_caller
; CHECK: br i1
; CHECK: br i1
; CHECK: call void @bar.2.
; LIMIT-LABEL: @dummy_caller
; LIMIT-NOT: br
; LIMIT: call i32 @bar(
  %tmp = tail call i32 @bar(i32 %arg)
  ret i32 %tmp
}

define i32 @bar_multi_ret(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb4, label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 (...) @channels() #1
  %tmp3 = icmp slt i32 %tmp2, %arg
  br i1 %tmp3, label %bb4, label %bb5

bb4:                                              ; preds = %bb1, %bb
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  %tmp4 = icmp slt i32 %arg, 10
  br i1 %tmp4, label %bb6, label %bb5
bb6:
  tail call void (...) @foo() #1
  %tmp5 = icmp slt i32 %arg, 3
  br i1 %tmp5, label %bb7, label %bb5
bb7:
  tail call void (...) @foo() #1
  br label %bb8
bb8:
  ret i32 0

bb5:                                              ; preds = %bb4, %bb1
  %.0 = phi i32 [ 0, %bb4 ], [ 1, %bb1 ], [0, %bb6]
  ret i32 %.0
}

define i32 @dummy_caller2(i32 %arg) local_unnamed_addr #0 {
; CHECK: br i1
; CHECK: br i1
; CHECK: call {{.*}} @bar_multi_ret.1.
  %tmp = tail call i32 @bar_multi_ret(i32 %arg)
  ret i32 %tmp
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 5.0.0 (trunk 300576)"}
