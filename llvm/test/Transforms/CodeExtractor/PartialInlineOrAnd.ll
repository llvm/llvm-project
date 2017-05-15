; RUN: opt < %s -partial-inliner -S | FileCheck %s
; RUN: opt < %s -passes=partial-inliner -S | FileCheck %s
; RUN: opt < %s -partial-inliner -max-num-inline-blocks=3 -skip-partial-inlining-cost-analysis  -S | FileCheck --check-prefix=LIMIT3 %s
; RUN: opt < %s -passes=partial-inliner -max-num-inline-blocks=3 -skip-partial-inlining-cost-analysis -S | FileCheck  --check-prefix=LIMIT3 %s
; RUN: opt < %s -partial-inliner -max-num-inline-blocks=2 -S | FileCheck --check-prefix=LIMIT2 %s
; RUN: opt < %s -passes=partial-inliner -max-num-inline-blocks=2 -S | FileCheck  --check-prefix=LIMIT2 %s


; Function Attrs: nounwind uwtable
define i32 @bar(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb4, label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 (...) @n() #2
  %tmp3 = icmp slt i32 %tmp2, %arg
  br i1 %tmp3, label %bb4, label %bb8

bb4:                                              ; preds = %bb1, %bb
  %tmp5 = tail call i32 (...) @m() #2
  %tmp6 = icmp sgt i32 %tmp5, %arg
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb4
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

bb8:                                              ; preds = %bb7, %bb4, %bb1
  %tmp9 = phi i32 [ 0, %bb7 ], [ 1, %bb4 ], [ 1, %bb1 ]
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
; LIMIT3-LABEL: @dummy_caller
; LIMIT3: br i1
; LIMIT3: br i1
; LIMIT3-NOT: br i1
; LIMIT3: call void @bar.1_
; LIMIT2-LABEL: @dummy_caller
; LIMIT2-NOT: br i1
; LIMIT2: call i32 @bar(
  %tmp = tail call i32 @bar(i32 %arg)
  ret i32 %tmp
}

attributes #0 = { nounwind } 
attributes #1 = { nounwind }
attributes #2 = { nounwind }

