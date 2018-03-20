; RUN: llc -march=hexagon < %s | FileCheck %s

; Make sure we generate a hardware loop and pipeline the inner loop using
; 4 packets, which is equivalent to the hand-coded version.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK: }
; CHECK: {
; CHECK-NOT: }
; CHECK: }{{[ \t]*}}:endloop0

define void @f0(i8* noalias %a0, i32 %a1, i32 %a2, i32 %a3, i8* noalias nocapture %a4, i32 %a5, i32 %a6) #0 {
b0:
  %v0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 8388736)
  %v1 = zext i32 %a3 to i64
  %v2 = shl nuw i64 %v1, 32
  %v3 = zext i32 %a1 to i64
  %v4 = shl nuw nsw i64 %v3, 16
  %v5 = or i64 %v4, %v2
  %v6 = or i64 %v5, 281474976710658
  tail call void asm sideeffect "    l2fetch($0, $1)\0A", "r,r"(i8* %a0, i64 %v6) #2, !srcloc !0
  %v7 = tail call i32 @llvm.hexagon.S2.ct0(i32 %a6)
  %v8 = add i32 %v7, 1
  %v9 = lshr i32 %a1, %v8
  %v10 = mul i32 %a6, 2
  %v11 = mul i32 %v10, %v9
  %v12 = sub i32 %a1, %v11
  %v13 = lshr i32 %v12, 1
  %v14 = tail call <512 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %v13)
  %v15 = icmp eq i32 %a2, 0
  br i1 %v15, label %b11, label %b1

b1:                                               ; preds = %b0
  %v16 = mul i32 %a3, 2
  %v17 = icmp eq i32 %v9, 0
  %v18 = icmp eq i32 %v11, %a1
  %v19 = icmp ugt i32 %v12, %a6
  %v20 = mul i32 %v9, 64
  %v21 = getelementptr i8, i8* %a4, i32 %v20
  %v22 = mul i32 %v9, 128
  %v23 = add i32 %v22, %a3
  %v24 = getelementptr i8, i8* %a0, i32 %v23
  %v25 = getelementptr i8, i8* %a0, i32 %v22
  br label %b2

b2:                                               ; preds = %b10, %b1
  %v26 = phi i8* [ %v25, %b1 ], [ %v90, %b10 ]
  %v27 = phi i8* [ %v24, %b1 ], [ %v89, %b10 ]
  %v28 = phi i8* [ %v21, %b1 ], [ %v88, %b10 ]
  %v29 = phi <16 x i32> [ undef, %b1 ], [ %v85, %b10 ]
  %v30 = phi <16 x i32> [ undef, %b1 ], [ %v84, %b10 ]
  %v31 = phi i8* [ %a0, %b1 ], [ %v86, %b10 ]
  %v32 = phi i8* [ %a4, %b1 ], [ %v87, %b10 ]
  %v33 = phi i32 [ 0, %b1 ], [ %v37, %b10 ]
  %v34 = bitcast i8* %v26 to <16 x i32>*
  %v35 = bitcast i8* %v27 to <16 x i32>*
  %v36 = bitcast i8* %v28 to <16 x i32>*
  %v37 = add nsw i32 %v33, 2
  %v38 = icmp ult i32 %v37, %a2
  br i1 %v38, label %b3, label %b4

b3:                                               ; preds = %b2
  %v39 = getelementptr inbounds i8, i8* %v31, i32 %v16
  tail call void asm sideeffect "    l2fetch($0, $1)\0A", "r,r"(i8* %v39, i64 %v6) #2, !srcloc !1
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v40 = bitcast i8* %v32 to <16 x i32>*
  %v41 = bitcast i8* %v31 to <16 x i32>*
  %v42 = getelementptr inbounds i8, i8* %v31, i32 %a3
  %v43 = bitcast i8* %v42 to <16 x i32>*
  br i1 %v17, label %b6, label %b5

b5:                                               ; preds = %b5, %b4
  %v44 = phi <16 x i32>* [ %v54, %b5 ], [ %v43, %b4 ]
  %v45 = phi <16 x i32>* [ %v52, %b5 ], [ %v41, %b4 ]
  %v46 = phi <16 x i32>* [ %v61, %b5 ], [ %v40, %b4 ]
  %v47 = phi i32 [ %v62, %b5 ], [ 0, %b4 ]
  %v48 = getelementptr inbounds <16 x i32>, <16 x i32>* %v45, i32 1
  %v49 = load <16 x i32>, <16 x i32>* %v45, align 64, !tbaa !2
  %v50 = getelementptr inbounds <16 x i32>, <16 x i32>* %v44, i32 1
  %v51 = load <16 x i32>, <16 x i32>* %v44, align 64, !tbaa !2
  %v52 = getelementptr inbounds <16 x i32>, <16 x i32>* %v45, i32 2
  %v53 = load <16 x i32>, <16 x i32>* %v48, align 64, !tbaa !2
  %v54 = getelementptr inbounds <16 x i32>, <16 x i32>* %v44, i32 2
  %v55 = load <16 x i32>, <16 x i32>* %v50, align 64, !tbaa !2
  %v56 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %v0, <16 x i32> %v49, i32 1077952576)
  %v57 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %v0, <16 x i32> %v53, i32 1077952576)
  %v58 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %v56, <16 x i32> %v51, i32 1077952576)
  %v59 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %v57, <16 x i32> %v55, i32 1077952576)
  %v60 = tail call <16 x i32> @llvm.hexagon.V6.vpackob(<16 x i32> %v59, <16 x i32> %v58)
  %v61 = getelementptr inbounds <16 x i32>, <16 x i32>* %v46, i32 1
  store <16 x i32> %v60, <16 x i32>* %v46, align 64, !tbaa !2
  %v62 = add nsw i32 %v47, 1
  %v63 = icmp eq i32 %v62, %v9
  br i1 %v63, label %b6, label %b5

b6:                                               ; preds = %b5, %b4
  %v64 = phi <16 x i32> [ %v29, %b4 ], [ %v55, %b5 ]
  %v65 = phi <16 x i32> [ %v30, %b4 ], [ %v53, %b5 ]
  %v66 = phi <16 x i32>* [ %v43, %b4 ], [ %v35, %b5 ]
  %v67 = phi <16 x i32>* [ %v41, %b4 ], [ %v34, %b5 ]
  %v68 = phi <16 x i32>* [ %v40, %b4 ], [ %v36, %b5 ]
  br i1 %v18, label %b10, label %b7

b7:                                               ; preds = %b6
  %v69 = load <16 x i32>, <16 x i32>* %v67, align 64, !tbaa !2
  %v70 = load <16 x i32>, <16 x i32>* %v66, align 64, !tbaa !2
  br i1 %v19, label %b8, label %b9

b8:                                               ; preds = %b7
  %v71 = getelementptr inbounds <16 x i32>, <16 x i32>* %v66, i32 1
  %v72 = getelementptr inbounds <16 x i32>, <16 x i32>* %v67, i32 1
  %v73 = load <16 x i32>, <16 x i32>* %v72, align 64, !tbaa !2
  %v74 = load <16 x i32>, <16 x i32>* %v71, align 64, !tbaa !2
  br label %b9

b9:                                               ; preds = %b8, %b7
  %v75 = phi <16 x i32> [ %v73, %b8 ], [ %v65, %b7 ]
  %v76 = phi <16 x i32> [ %v74, %b8 ], [ %v64, %b7 ]
  %v77 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %v0, <16 x i32> %v69, i32 1077952576)
  %v78 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %v0, <16 x i32> %v75, i32 1077952576)
  %v79 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %v77, <16 x i32> %v70, i32 1077952576)
  %v80 = tail call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %v78, <16 x i32> %v76, i32 1077952576)
  %v81 = tail call <16 x i32> @llvm.hexagon.V6.vpackob(<16 x i32> %v80, <16 x i32> %v79)
  %v82 = load <16 x i32>, <16 x i32>* %v68, align 64, !tbaa !2
  %v83 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v14, <16 x i32> %v81, <16 x i32> %v82)
  store <16 x i32> %v83, <16 x i32>* %v68, align 64, !tbaa !2
  br label %b10

b10:                                              ; preds = %b9, %b6
  %v84 = phi <16 x i32> [ %v75, %b9 ], [ %v65, %b6 ]
  %v85 = phi <16 x i32> [ %v76, %b9 ], [ %v64, %b6 ]
  %v86 = getelementptr inbounds i8, i8* %v31, i32 %v16
  %v87 = getelementptr inbounds i8, i8* %v32, i32 %a5
  %v88 = getelementptr i8, i8* %v28, i32 %a5
  %v89 = getelementptr i8, i8* %v27, i32 %v16
  %v90 = getelementptr i8, i8* %v26, i32 %v16
  br i1 %v38, label %b2, label %b11

b11:                                              ; preds = %b10, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.ct0(i32) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.scalar2(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackob(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1>, <16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!0 = !{i32 -2146401371}
!1 = !{i32 -2146401153}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
