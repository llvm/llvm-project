; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that we generate a .cur

; CHECK: v{{[0-9]*}}.cur

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, i32 %a1, i32 %a2, ptr %a3, ptr %a4) #0 {
b0:
  br i1 undef, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi ptr [ %a0, %b1 ], [ %v4, %b2 ]
  %v1 = phi i32 [ 0, %b1 ], [ %v23, %b2 ]
  %v2 = phi <16 x i32> [ zeroinitializer, %b1 ], [ %v6, %b2 ]
  %v3 = phi <16 x i32> [ zeroinitializer, %b1 ], [ zeroinitializer, %b2 ]
  %v4 = getelementptr inbounds i8, ptr %v0, i32 64
  %v6 = load <16 x i32>, ptr %v4, align 64, !tbaa !0
  %v7 = load <16 x i32>, ptr %a3, align 64, !tbaa !0
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v6, <16 x i32> %v2, i32 4)
  %v9 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> zeroinitializer, <16 x i32> %v3, i32 4)
  %v10 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v7, <16 x i32> zeroinitializer, i32 4)
  %v11 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v8, <16 x i32> %v2)
  %v12 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v10, <16 x i32> zeroinitializer)
  %v13 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %v11, i32 0, i32 0)
  %v14 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v13, <32 x i32> zeroinitializer, i32 undef, i32 0)
  %v15 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %v14, <32 x i32> undef, i32 undef, i32 0)
  %v16 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v15)
  %v17 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %v16, <16 x i32> undef, i32 %a1)
  %v18 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> undef, <16 x i32> %v17)
  store <16 x i32> %v18, ptr %a3, align 64, !tbaa !0
  %v19 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> zeroinitializer, <32 x i32> %v12, i32 undef, i32 1)
  %v20 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v19)
  %v21 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %v20, <16 x i32> undef, i32 %a1)
  %v22 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v21, <16 x i32> undef)
  store <16 x i32> %v22, ptr %a4, align 64, !tbaa !0
  %v23 = add nsw i32 %v1, 64
  %v24 = icmp slt i32 %v23, %a2
  br i1 %v24, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32>, i32, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32>, <32 x i32>, i32, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32>, <16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
