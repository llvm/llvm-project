; RUN: opt -S  -partial-inliner -max-num-inline-blocks=2 -skip-partial-inlining-cost-analysis < %s  | FileCheck %s
; RUN: opt -S -passes=partial-inliner -max-num-inline-blocks=2 -skip-partial-inlining-cost-analysis < %s  | FileCheck %s
define i32 @test(i32 %arg) local_unnamed_addr #0 {
bb:
  %tmp = tail call i32 (...) @bar() #1
  %tmp1 = icmp slt i32 %arg, 0
  br i1 %tmp1, label %bb6, label %bb2

bb2:                                              ; preds = %bb
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  %tmp3 = tail call i32 (...) @bar() #1
  %tmp4 = icmp eq i32 %tmp3, 10
  br i1 %tmp4, label %bb6, label %bb5

bb5:                                              ; preds = %bb2
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  tail call void (...) @foo() #1
  br label %bb6

bb6:                                              ; preds = %bb5, %bb2, %bb
  %tmp7 = phi i32 [ %tmp, %bb5 ], [ 0, %bb ], [ %tmp, %bb2 ]
  ret i32 %tmp7
}

declare i32 @bar(...) local_unnamed_addr #1

declare void @foo(...) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define i32 @dummy_caller(i32 %arg) local_unnamed_addr #0 {
; CHECK-LABEL: @dummy_caller
; CHECK: codeRepl.i:
; CHECK:  call void @test.1.bb2()
; CHECK-NOT: load
; CHECK:  br

bb:
  %tmp = tail call i32 @test(i32 %arg)
  ret i32 %tmp
}

; CHECK-LABEL: define internal void @test.1.bb2()
; CHECK: .exitStub:
; CHECK-NOT:  store i32 %tmp7, i32* %tmp7.out
; CHECK: ret


attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind uwtable }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 303574)"}
