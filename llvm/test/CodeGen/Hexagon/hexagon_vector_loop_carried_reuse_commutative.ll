; RUN: opt < %s -march=hexagon -hexagon-vlcr | opt -passes=adce -S | FileCheck %s

; CHECK: %v32.hexagon.vlcr = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, ptr noalias nocapture %a1, i32 %a2) local_unnamed_addr #0 {
b0:
  %v0 = getelementptr inbounds i8, ptr %a0, i32 %a2
  %v1 = mul nsw i32 %a2, 2
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i32, ptr @g0, align 4, !tbaa !0
  %v4 = icmp sgt i32 %v3, 0
  br i1 %v4, label %b1, label %b4

b1:                                               ; preds = %b0
  %v6 = load <32 x i32>, ptr %v2, align 128, !tbaa !4
  %v7 = getelementptr inbounds i8, ptr %v2, i32 128
  %v10 = load <32 x i32>, ptr %v0, align 128, !tbaa !4
  %v11 = getelementptr inbounds i8, ptr %v0, i32 128
  %v14 = load <32 x i32>, ptr %a0, align 128, !tbaa !4
  %v15 = getelementptr inbounds i8, ptr %a0, i32 128
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v18 = phi ptr [ %a1, %b1 ], [ %v37, %b2 ]
  %v19 = phi ptr [ %v7, %b1 ], [ %v30, %b2 ]
  %v20 = phi ptr [ %v11, %b1 ], [ %v28, %b2 ]
  %v21 = phi ptr [ %v15, %b1 ], [ %v26, %b2 ]
  %v22 = phi i32 [ 0, %b1 ], [ %v38, %b2 ]
  %v23 = phi <32 x i32> [ %v14, %b1 ], [ %v27, %b2 ]
  %v24 = phi <32 x i32> [ %v10, %b1 ], [ %v29, %b2 ]
  %v25 = phi <32 x i32> [ %v6, %b1 ], [ %v31, %b2 ]
  %v26 = getelementptr inbounds <32 x i32>, ptr %v21, i32 1
  %v27 = load <32 x i32>, ptr %v21, align 128, !tbaa !4
  %v28 = getelementptr inbounds <32 x i32>, ptr %v20, i32 1
  %v29 = load <32 x i32>, ptr %v20, align 128, !tbaa !4
  %v30 = getelementptr inbounds <32 x i32>, ptr %v19, i32 1
  %v31 = load <32 x i32>, ptr %v19, align 128, !tbaa !4
  %v32 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v23, <32 x i32> %v24)
  %v33 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v32, <32 x i32> %v25)
  %v34 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v29, <32 x i32> %v27)
  %v35 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v34, <32 x i32> %v31)
  %v36 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> %v35, <32 x i32> %v33, i32 1)
  %v37 = getelementptr inbounds <32 x i32>, ptr %v18, i32 1
  store <32 x i32> %v36, ptr %v18, align 128, !tbaa !4
  %v38 = add nuw nsw i32 %v22, 128
  %v39 = icmp slt i32 %v38, %v3
  br i1 %v39, label %b2, label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b,-long-calls" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
