; RUN: opt < %s -partial-inliner -S  | FileCheck %s
; RUN: opt < %s -passes=partial-inliner -S  | FileCheck %s


; Function Attrs: nounwind
declare void @foo(...) local_unnamed_addr #0

; Function Attrs: noinline
define i32 @caller(i32 (i32)* nocapture %arg, i32 (i32)* nocapture %arg1, i32 %arg2) local_unnamed_addr #1 {
bb:
  %tmp = tail call i32 %arg(i32 %arg2) #0
  %tmp3 = tail call i32 %arg1(i32 %arg2) #0
  %tmp4 = add nsw i32 %tmp3, %tmp
  ret i32 %tmp4
}

; Function Attrs: nounwind
define i32 @bar(i32 %arg) #0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  tail call void (...) @foo() #0
  tail call void (...) @foo() #0
  tail call void (...) @foo() #0
  tail call void (...) @foo() #0
  tail call void (...) @foo() #0
  tail call void (...) @foo() #0
  tail call void (...) @foo() #0
  tail call void (...) @foo() #0
  tail call void (...) @foo() #0
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %tmp3 = phi i32 [ 0, %bb1 ], [ 1, %bb ]
  ret i32 %tmp3
}

; Function Attrs: nounwind
define i32 @dummy_caller(i32 %arg) local_unnamed_addr #0 {
bb:
; CHECK-LABEL: @dummy_caller
; check that caller is not wrongly inlined by partial inliner
; CHECK: call i32 @caller
; CHECK-NOT: call .* @bar
  %tmp = tail call i32 @caller(i32 (i32)* nonnull @bar, i32 (i32)* nonnull @bar, i32 %arg)
  ret i32 %tmp
}

attributes #0 = { nounwind }
attributes #1 = { noinline }

!llvm.ident = !{!0}

!0 = !{!"clang version 5.0.0 (trunk 300897) (llvm/trunk 300947)"}
