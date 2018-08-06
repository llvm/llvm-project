; RUN: llc -march=hexagon -enable-pipeliner=false < %s | FileCheck %s

; Test that the vsplat and vmemu are not all serialized due to chain edges
; caused by the hasSideEffects flag. The exact code generation may change
; due to the scheduling changes, but we shouldn't see a series of
; vsplat and vmemu instructions that each occur in a single packet.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: vsplat
; CHECK-NEXT: vsplat
; CHECK: vsplat
; CHECK-NEXT: vsplat
; CHECK: endloop0

@g0 = global [256 x i8] c"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^00226644,,..**8888::66,,,,&&^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^22000022..4444>>::8888**..^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<<66220000226644<<>>::^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>><<446622000022>>", align 64

; Function Attrs: nounwind
define void @f0(i16** noalias nocapture readonly %a0, i16* noalias nocapture readonly %a1, i32* noalias nocapture %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6) #0 {
b0:
  %v0 = load <16 x i32>, <16 x i32>* bitcast ([256 x i8]* @g0 to <16 x i32>*), align 64, !tbaa !0
  %v1 = load <16 x i32>, <16 x i32>* bitcast (i8* getelementptr inbounds ([256 x i8], [256 x i8]* @g0, i32 0, i32 64) to <16 x i32>*), align 64, !tbaa !0
  %v2 = load <16 x i32>, <16 x i32>* bitcast (i8* getelementptr inbounds ([256 x i8], [256 x i8]* @g0, i32 0, i32 128) to <16 x i32>*), align 64, !tbaa !0
  %v3 = load <16 x i32>, <16 x i32>* bitcast (i8* getelementptr inbounds ([256 x i8], [256 x i8]* @g0, i32 0, i32 192) to <16 x i32>*), align 64, !tbaa !0
  %v4 = icmp sgt i32 %a5, 0
  br i1 %v4, label %b1, label %b5

b1:                                               ; preds = %b0
  %v5 = bitcast i32* %a2 to <16 x i32>*
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.vd0()
  %v7 = bitcast i16* %a1 to i64*
  %v8 = mul nsw i32 %a3, 4
  %v9 = add i32 %v8, %a6
  %v10 = add i32 %v9, 32
  %v11 = add i32 %a5, -1
  br label %b2

b2:                                               ; preds = %b4, %b1
  %v12 = phi i32 [ 0, %b1 ], [ %v59, %b4 ]
  %v13 = phi <16 x i32>* [ %v5, %b1 ], [ %v58, %b4 ]
  %v14 = getelementptr i16*, i16** %a0, i32 %v12
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v15 = phi i16** [ %v14, %b2 ], [ %v57, %b3 ]
  %v16 = phi i32 [ 0, %b2 ], [ %v55, %b3 ]
  %v17 = phi i64* [ %v7, %b2 ], [ %v23, %b3 ]
  %v18 = phi <16 x i32> [ %v6, %b2 ], [ %v54, %b3 ]
  %v19 = load i16*, i16** %v15, align 4, !tbaa !3
  %v20 = getelementptr inbounds i16, i16* %v19, i32 %v9
  %v21 = getelementptr inbounds i64, i64* %v17, i32 1
  %v22 = load i64, i64* %v17, align 8, !tbaa !0
  %v23 = getelementptr inbounds i64, i64* %v17, i32 2
  %v24 = load i64, i64* %v21, align 8, !tbaa !0
  %v25 = trunc i64 %v22 to i32
  %v26 = lshr i64 %v22, 32
  %v27 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v25)
  %v28 = trunc i64 %v26 to i32
  %v29 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v28)
  %v30 = trunc i64 %v24 to i32
  %v31 = lshr i64 %v24, 32
  %v32 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v30)
  %v33 = trunc i64 %v31 to i32
  %v34 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v33)
  %v35 = bitcast i16* %v20 to <16 x i32>*
  %v36 = load <16 x i32>, <16 x i32>* %v35, align 4, !tbaa !0
  %v37 = getelementptr inbounds i16, i16* %v19, i32 %v10
  %v38 = bitcast i16* %v37 to <16 x i32>*
  %v39 = load <16 x i32>, <16 x i32>* %v38, align 4, !tbaa !0
  %v40 = tail call <16 x i32> @llvm.hexagon.V6.vpackeh(<16 x i32> %v39, <16 x i32> %v36)
  %v41 = tail call <16 x i32> @llvm.hexagon.V6.vpackeh(<16 x i32> %v40, <16 x i32> %v40)
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32> %v41, <16 x i32> %v0)
  %v43 = tail call <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32> %v41, <16 x i32> %v1)
  %v44 = tail call <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32> %v41, <16 x i32> %v2)
  %v45 = tail call <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32> %v41, <16 x i32> %v3)
  %v46 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v27, <16 x i32> %v42)
  %v47 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v29, <16 x i32> %v43)
  %v48 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v32, <16 x i32> %v44)
  %v49 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v34, <16 x i32> %v45)
  %v50 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32> %v46, <16 x i32> %v46)
  %v51 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc(<16 x i32> %v50, <16 x i32> %v47, <16 x i32> %v47)
  %v52 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc(<16 x i32> %v51, <16 x i32> %v48, <16 x i32> %v48)
  %v53 = tail call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc(<16 x i32> %v52, <16 x i32> %v49, <16 x i32> %v49)
  %v54 = tail call <16 x i32> @llvm.hexagon.V6.vasrw.acc(<16 x i32> %v18, <16 x i32> %v53, i32 6)
  %v55 = add nsw i32 %v16, 1
  %v56 = icmp eq i32 %v16, 7
  %v57 = getelementptr i16*, i16** %v15, i32 1
  br i1 %v56, label %b4, label %b3

b4:                                               ; preds = %b3
  %v58 = getelementptr inbounds <16 x i32>, <16 x i32>* %v13, i32 1
  store <16 x i32> %v54, <16 x i32>* %v13, align 64, !tbaa !0
  %v59 = add nsw i32 %v12, 1
  %v60 = icmp eq i32 %v12, %v11
  br i1 %v60, label %b5, label %b2

b5:                                               ; preds = %b4, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vd0() #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackeh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrw.acc(<16 x i32>, <16 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !1, i64 0}
