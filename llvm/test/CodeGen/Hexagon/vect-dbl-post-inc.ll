; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that we generate a post-increment when using double hvx (128B)
; post-increment operations.

; CHECK: = vmem(r{{[0-9]+}}++#1)
; CHECK: vmem(r{{[0-9]+}}++#1)

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, ptr noalias nocapture %a1, i32 %a2) #0 {
b0:
  %v0 = icmp sgt i32 %a2, 0
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v3 = phi ptr [ %v9, %b2 ], [ %a0, %b1 ]
  %v4 = phi ptr [ %v10, %b2 ], [ %a1, %b1 ]
  %v5 = phi i32 [ %v7, %b2 ], [ 0, %b1 ]
  %v6 = load <32 x i32>, ptr %v3, align 128, !tbaa !0
  store <32 x i32> %v6, ptr %v4, align 128, !tbaa !0
  %v7 = add nsw i32 %v5, 1
  %v8 = icmp eq i32 %v7, %a2
  %v9 = getelementptr <32 x i32>, ptr %v3, i32 1
  %v10 = getelementptr <32 x i32>, ptr %v4, i32 1
  br i1 %v8, label %b3, label %b2

b3:                                               ; preds = %b2, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
