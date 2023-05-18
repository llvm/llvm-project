; RUN: opt -S < %s -passes=partial-inliner -partial-inlining-extra-penalty=2000 | FileCheck %s
define i32 @outline_region_notlikely(ptr %arg) local_unnamed_addr {
bb:
;  ptr != null is predicted to be true
  %tmp = icmp ne ptr %arg, null
  br i1 %tmp, label %bb8, label %bb1, !prof !2

; bb1 is not likely
bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 @foo(ptr nonnull %arg)
  %tmp3 = tail call i32 @foo(ptr nonnull %arg)
  %tmp4 = tail call i32 @foo(ptr nonnull %arg)
  %tmp5 = tail call i32 @foo(ptr nonnull %arg)
  %tmp6 = tail call i32 @foo(ptr nonnull %arg)
  %tmp7 = tail call i32 @foo(ptr nonnull %arg)
  br label %bb8

bb8:                                              ; preds = %bb1, %bb
  %tmp9 = phi i32 [ 0, %bb1 ], [ 1, %bb ]
  ret i32 %tmp9
}

define i32 @dummy_caller(ptr %arg) local_unnamed_addr {
; CHECK-LABEL: @dummy_caller
  %tmp = call i32 @outline_region_notlikely(ptr %arg)
  ret i32 %tmp
 }


; CHECK-LABEL: define internal void @outline_region_notlikely.1.bb1(ptr %arg) {
; CHECK-NEXT: newFuncRoot:

declare i32 @foo(ptr %arg)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 304489)"}
!2 = !{!"branch_weights", i32 2000, i32 1}
