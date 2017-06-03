; RUN: opt -S < %s  -partial-inliner -partial-inlining-extra-penalty=10 | FileCheck %s
; RUN: opt -S < %s  -passes=partial-inliner -partial-inlining-extra-penalty=10 | FileCheck %s
define i32 @outline_region_notlikely(i32* %arg) local_unnamed_addr {
bb:
;  ptr != null is predicted to be true 
  %tmp = icmp ne i32* %arg, null
  br i1 %tmp, label %bb8, label %bb1

; bb1 is not likely
bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 @foo(i32* nonnull %arg)
  %tmp3 = tail call i32 @foo(i32* nonnull %arg)
  %tmp4 = tail call i32 @foo(i32* nonnull %arg)
  %tmp5 = tail call i32 @foo(i32* nonnull %arg)
  %tmp6 = tail call i32 @foo(i32* nonnull %arg)
  %tmp7 = tail call i32 @foo(i32* nonnull %arg)
  br label %bb8

bb8:                                              ; preds = %bb1, %bb
  %tmp9 = phi i32 [ 0, %bb1 ], [ 1, %bb ]
  ret i32 %tmp9
}

define i32 @outline_region_likely(i32* %arg) local_unnamed_addr {
bb:
;  ptr == null is predicted to be false
  %tmp = icmp eq i32* %arg, null
  br i1 %tmp, label %bb8, label %bb1

; bb1 is likely
bb1:                                              ; preds = %bb
  %tmp2 = tail call i32 @foo(i32* nonnull %arg)
  %tmp3 = tail call i32 @foo(i32* nonnull %arg)
  %tmp4 = tail call i32 @foo(i32* nonnull %arg)
  %tmp5 = tail call i32 @foo(i32* nonnull %arg)
  %tmp6 = tail call i32 @foo(i32* nonnull %arg)
  %tmp7 = tail call i32 @foo(i32* nonnull %arg)
  br label %bb8

bb8:                                              ; preds = %bb1, %bb
  %tmp9 = phi i32 [ 0, %bb1 ], [ 1, %bb ]
  ret i32 %tmp9
}

declare i32 @foo(i32* %arg)

define i32 @dummy_caller(i32* %arg) local_unnamed_addr {
; CHECK-LABEL: @dummy_caller
  %tmp = call i32 @outline_region_notlikely(i32* %arg)
; CHECK:  call void @outline_region_notlikely.2_bb1
  %tmp2 = tail call i32 @outline_region_likely(i32* %arg)
; CHECK: %tmp2 = tail call i32 @outline_region_likely(i32* %arg)
  ret i32 %tmp

}

; CHECK-LABEL: define internal void @outline_region_notlikely.2_bb1(i32* %arg) {
; CHECK-NEXT: newFuncRoot:

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 304489)"}
