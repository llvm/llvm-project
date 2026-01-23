; RUN: llc -mtriple=hexagon -O0 < %s | FileCheck %s

; CHECK: v{{[0-9]+}}:{{[0-9]+}} = vcombine(v{{[0-9]+}},v{{[0-9]+}})

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr nocapture readnone %a0, i32 %a1, i32 %a2, i32 %a3, ptr nocapture %a4, i32 %a5) #0 {
b0:
  %v1 = mul i32 %a5, -2
  %v2 = add i32 %v1, %a1
  %v3 = and i32 %a5, 63
  %v4 = add i32 %v2, %v3
  %v5 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 -1)
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %v7 = tail call <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %v4)
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32> %v6, <64 x i1> %v7, i32 12)
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v8, <16 x i32> %v8)
  %v10 = and i32 %v4, 511
  %v11 = icmp eq i32 %v10, 0
  br i1 %v11, label %b1, label %b2

b1:                                               ; preds = %b0
  %v12 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v5, <16 x i32> %v8)
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v13 = phi <32 x i32> [ %v12, %b1 ], [ %v9, %b0 ]
  %v14 = icmp sgt i32 %v4, 0
  br i1 %v14, label %b3, label %b6

b3:                                               ; preds = %b2
  %v15 = tail call <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %a5)
  %v16 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v15, i32 16843009)
  %v17 = tail call <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32> %v16)
  %v18 = add i32 %v3, %a1
  %v19 = add i32 %v18, -1
  %v20 = add i32 %v19, %v1
  %v21 = lshr i32 %v20, 9
  %v22 = mul i32 %v21, 16
  %v23 = add nuw nsw i32 %v22, 16
  %v24 = getelementptr i32, ptr %a4, i32 %v23
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v25 = phi i32 [ %v4, %b3 ], [ %v30, %b4 ]
  %v26 = phi <16 x i32> [ %v17, %b3 ], [ %v5, %b4 ]
  %v27 = phi ptr [ %a4, %b3 ], [ %v29, %b4 ]
  %v28 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> undef, <16 x i32> %v26)
  %v29 = getelementptr inbounds <16 x i32>, ptr %v27, i32 1
  store <16 x i32> %v28, ptr %v27, align 64, !tbaa !0
  %v30 = add nsw i32 %v25, -512
  %v31 = icmp sgt i32 %v30, 0
  br i1 %v31, label %b4, label %b5

b5:                                               ; preds = %b4
  br label %b6

b6:                                               ; preds = %b5, %b2
  %v33 = phi ptr [ %v24, %b5 ], [ %a4, %b2 ]
  %v34 = load <16 x i32>, ptr %v33, align 64, !tbaa !0
  %v35 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v13)
  %v36 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> %v34, <16 x i32> %v35)
  store <16 x i32> %v36, ptr %v33, align 64, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32>, <64 x i1>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vand(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
