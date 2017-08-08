; The outlined region has high frequency  and the outlining
; call sequence is expensive (input, output, multiple exit etc)
; RUN: opt < %s -partial-inliner -max-num-inline-blocks=2 -S | FileCheck %s
; RUN: opt < %s -passes=partial-inliner -max-num-inline-blocks=2 -S | FileCheck %s
; RUN: opt < %s -partial-inliner -skip-partial-inlining-cost-analysis -max-num-inline-blocks=2 -S | FileCheck --check-prefix=NOCOST %s
; RUN: opt < %s -passes=partial-inliner -skip-partial-inlining-cost-analysis -max-num-inline-blocks=2 -S | FileCheck  --check-prefix=NOCOST %s


; Function Attrs: nounwind
define i32 @bar_hot_outline_region(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb16, !prof !1

bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 (...) @foo() #0
  %tmp3 = tail call i32 (...) @foo() #0
  %tmp4 = tail call i32 (...) @foo() #0
  %tmp5 = tail call i32 (...) @foo() #0
  %tmp6 = tail call i32 (...) @foo() #0
  %tmp7 = tail call i32 (...) @foo() #0
  %tmp8 = add nsw i32 %arg, 1
  %tmp9 = tail call i32 @goo(i32 %tmp8) #0
  %tmp10 = tail call i32 (...) @foo() #0
  %tmp11 = icmp eq i32 %tmp10, 0
  br i1 %tmp11, label %bb12, label %bb16

bb12:                                             ; preds = %bb1
  %tmp13 = tail call i32 (...) @foo() #0
  %tmp14 = icmp eq i32 %tmp13, 0
  %tmp15 = select i1 %tmp14, i32 0, i32 3
  br label %bb16

bb16:                                             ; preds = %bb12, %bb1, %bb
  %tmp17 = phi i32 [ 2, %bb1 ], [ %tmp15, %bb12 ], [ 0, %bb ]
  ret i32 %tmp17
}

define i32 @bar_cold_outline_region(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = icmp slt i32 %arg, 0
  br i1 %tmp, label %bb1, label %bb16, !prof !2

bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 (...) @foo() #0
  %tmp3 = tail call i32 (...) @foo() #0
  %tmp4 = tail call i32 (...) @foo() #0
  %tmp5 = tail call i32 (...) @foo() #0
  %tmp6 = tail call i32 (...) @foo() #0
  %tmp7 = tail call i32 (...) @foo() #0
  %tmp8 = add nsw i32 %arg, 1
  %tmp9 = tail call i32 @goo(i32 %tmp8) #0
  %tmp10 = tail call i32 (...) @foo() #0
  %tmp11 = icmp eq i32 %tmp10, 0
  br i1 %tmp11, label %bb12, label %bb16

bb12:                                             ; preds = %bb1
  %tmp13 = tail call i32 (...) @foo() #0
  %tmp14 = icmp eq i32 %tmp13, 0
  %tmp15 = select i1 %tmp14, i32 0, i32 3
  br label %bb16

bb16:                                             ; preds = %bb12, %bb1, %bb
  %tmp17 = phi i32 [ 2, %bb1 ], [ %tmp15, %bb12 ], [ 0, %bb ]
  ret i32 %tmp17
}

; Function Attrs: nounwind
declare i32 @foo(...) local_unnamed_addr #0

; Function Attrs: nounwind
declare i32 @goo(i32) local_unnamed_addr #0

; Function Attrs: nounwind
define i32 @dummy_caller(i32 %arg) local_unnamed_addr #0 {
bb:
; CHECK-LABEL: @dummy_caller
; CHECK-NOT: br i1
; CHECK-NOT: call{{.*}}bar_hot_outline_region. 
; NOCOST-LABEL: @dummy_caller
; NOCOST: br i1
; NOCOST: call{{.*}}bar_hot_outline_region.

  %tmp = tail call i32 @bar_hot_outline_region(i32 %arg)
  ret i32 %tmp
}

define i32 @dummy_caller2(i32 %arg) local_unnamed_addr #0 {
bb:
; CHECK-LABEL: @dummy_caller2
; CHECK: br i1
; CHECK: call{{.*}}bar_cold_outline_region.
; NOCOST-LABEL: @dummy_caller2
; NOCOST: br i1
; NOCOST: call{{.*}}bar_cold_outline_region.

  %tmp = tail call i32 @bar_cold_outline_region(i32 %arg)
  ret i32 %tmp
}

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 5.0.0 (trunk 301898)"}
!1 = !{!"branch_weights", i32 2000, i32 1}
!2 = !{!"branch_weights", i32 1, i32 100}
