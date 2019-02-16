; RUN: opt < %s -partial-inliner -S | FileCheck %s
; RUN: opt < %s -passes=partial-inliner -S | FileCheck %s
; RUN: opt < %s -partial-inliner -max-num-inline-blocks=3 -S | FileCheck --check-prefix=LIMIT %s
; RUN: opt < %s -passes=partial-inliner -max-num-inline-blocks=3 -S | FileCheck  --check-prefix=LIMIT %s

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb4

bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 (...) @n() #2
  %tmp3 = icmp slt i32 %tmp2, %arg
  br i1 %tmp3, label %bb7, label %bb4

bb4:                                              ; preds = %bb1, %bb
  %tmp5 = tail call i32 (...) @m() #2
  %tmp6 = icmp slt i32 %tmp5, %arg
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb4, %bb1
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  tail call void (...) @foo() #2
  br label %bb8

bb8:                                              ; preds = %bb7, %bb4
  %tmp9 = phi i32 [ 0, %bb7 ], [ 1, %bb4 ]
  ret i32 %tmp9
}

declare i32 @n(...) local_unnamed_addr #1

declare i32 @m(...) local_unnamed_addr #1

declare void @foo(...) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define i32 @dummy_caller(i32 %arg) local_unnamed_addr #0 {
bb:
; CHECK-LABEL: @dummy_caller
; CHECK: br i1
; CHECK: br i1
; CHECK: br i1
; CHECK: call void @bar.1_
; LIMIT-LABEL: @dummy_caller
; LIMIT-NOT: br i1
; LIMIT: call i32 @bar
  %tmp = tail call i32 @bar(i32 %arg)
  ret i32 %tmp
}

attributes #0 = { nounwind } 
attributes #1 = { nounwind }
attributes #2 = { nounwind }

