; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; Ensure that the second use of ##grcolor doesn't get replaced with
; r26 which is an induction variable

; CHECK: r{{[0-9]+}} = ##g4
; CHECK: r{{[0-9]+}} = {{.*}}##g4

target triple = "hexagon-unknown--elf"

@g0 = external global [450 x i32]
@g1 = external global [842 x i32]
@g2 = external global [750 x i32]
@g3 = external global [750 x i32]
@g4 = external global [750 x i32]
@g5 = external global [750 x i32]
@g6 = external global [750 x i32]
@g7 = external global [750 x i32]
@g8 = external global [750 x i32]
@g9 = external global [750 x i32]
@g10 = external global i32
@g11 = external global [0 x i32]
@g12 = external global [0 x i32]

; Function Attrs: nounwind readonly
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = load i32, i32* @g10, align 4, !tbaa !0
  %v1 = icmp sgt i32 %v0, 0
  br i1 %v1, label %b1, label %b21

b1:                                               ; preds = %b0
  %v2 = getelementptr inbounds [842 x i32], [842 x i32]* @g1, i32 0, i32 %a0
  br label %b2

b2:                                               ; preds = %b19, %b1
  %v3 = phi i32 [ 0, %b1 ], [ %v79, %b19 ]
  %v4 = phi i32 [ 32767, %b1 ], [ %v78, %b19 ]
  %v5 = phi i32 [ 0, %b1 ], [ %v77, %b19 ]
  %v6 = phi i32 [ 0, %b1 ], [ %v76, %b19 ]
  %v7 = phi i32 [ 0, %b1 ], [ %v80, %b19 ]
  %v8 = getelementptr inbounds [750 x i32], [750 x i32]* @g5, i32 0, i32 %v7
  %v9 = load i32, i32* %v8, align 4, !tbaa !0
  %v10 = icmp eq i32 %v9, 0
  br i1 %v10, label %b19, label %b3

b3:                                               ; preds = %b2
  %v11 = getelementptr inbounds [750 x i32], [750 x i32]* @g4, i32 0, i32 %v7
  %v12 = load i32, i32* %v11, align 4, !tbaa !0
  %v13 = load i32, i32* %v2, align 4, !tbaa !0
  %v14 = getelementptr inbounds [750 x i32], [750 x i32]* @g4, i32 0, i32 %v13
  %v15 = load i32, i32* %v14, align 4, !tbaa !0
  %v16 = icmp eq i32 %v12, %v15
  br i1 %v16, label %b4, label %b8

b4:                                               ; preds = %b3
  %v17 = getelementptr inbounds [750 x i32], [750 x i32]* @g6, i32 0, i32 %v7
  %v18 = load i32, i32* %v17, align 4, !tbaa !0
  %v19 = icmp eq i32 %v18, 25
  br i1 %v19, label %b5, label %b19

b5:                                               ; preds = %b4
  %v20 = getelementptr inbounds [750 x i32], [750 x i32]* @g2, i32 0, i32 %v7
  %v21 = load i32, i32* %v20, align 4, !tbaa !0
  %v22 = icmp slt i32 %v21, 19
  br i1 %v22, label %b6, label %b19

b6:                                               ; preds = %b5
  %v23 = getelementptr inbounds [750 x i32], [750 x i32]* @g9, i32 0, i32 %v7
  %v24 = load i32, i32* %v23, align 4, !tbaa !0
  %v25 = icmp eq i32 %v24, 0
  br i1 %v25, label %b19, label %b7

b7:                                               ; preds = %b6
  %v26 = getelementptr inbounds [750 x i32], [750 x i32]* @g8, i32 0, i32 %v7
  %v27 = load i32, i32* %v26, align 4, !tbaa !0
  %v28 = mul nsw i32 %v27, 50
  %v29 = add nsw i32 %v28, %v3
  br label %b19

b8:                                               ; preds = %b3
  %v30 = getelementptr inbounds [750 x i32], [750 x i32]* @g9, i32 0, i32 %v7
  %v31 = load i32, i32* %v30, align 4, !tbaa !0
  %v32 = icmp eq i32 %v31, 0
  br i1 %v32, label %b13, label %b9

b9:                                               ; preds = %b8
  %v33 = getelementptr inbounds [750 x i32], [750 x i32]* @g7, i32 0, i32 %v7
  %v34 = load i32, i32* %v33, align 4, !tbaa !0
  %v35 = icmp eq i32 %v34, 0
  br i1 %v35, label %b10, label %b13

b10:                                              ; preds = %b9
  %v36 = getelementptr inbounds [750 x i32], [750 x i32]* @g6, i32 0, i32 %v7
  %v37 = load i32, i32* %v36, align 4, !tbaa !0
  %v38 = icmp slt i32 %v37, 18
  br i1 %v38, label %b11, label %b13

b11:                                              ; preds = %b10
  %v39 = getelementptr inbounds [0 x i32], [0 x i32]* @g11, i32 0, i32 %v37
  %v40 = load i32, i32* %v39, align 4, !tbaa !0
  %v41 = add nsw i32 %v40, 50
  %v42 = getelementptr inbounds [750 x i32], [750 x i32]* @g8, i32 0, i32 %v7
  %v43 = load i32, i32* %v42, align 4, !tbaa !0
  %v44 = mul nsw i32 %v41, %v43
  %v45 = icmp slt i32 %v44, %v4
  br i1 %v45, label %b12, label %b19

b12:                                              ; preds = %b11
  br label %b19

b13:                                              ; preds = %b10, %b9, %b8
  %v46 = getelementptr inbounds [750 x i32], [750 x i32]* @g2, i32 0, i32 %v7
  %v47 = load i32, i32* %v46, align 4, !tbaa !0
  %v48 = and i32 %v47, 31
  %v49 = getelementptr inbounds [0 x i32], [0 x i32]* @g12, i32 0, i32 %v48
  %v50 = load i32, i32* %v49, align 4, !tbaa !0
  %v51 = icmp eq i32 %v50, 0
  br i1 %v51, label %b19, label %b14

b14:                                              ; preds = %b13
  %v52 = getelementptr inbounds [750 x i32], [750 x i32]* @g2, i32 0, i32 %v13
  %v53 = load i32, i32* %v52, align 4, !tbaa !0
  %v54 = icmp slt i32 %v53, 11
  br i1 %v54, label %b15, label %b19

b15:                                              ; preds = %b14
  %v55 = getelementptr inbounds [750 x i32], [750 x i32]* @g6, i32 0, i32 %v7
  %v56 = load i32, i32* %v55, align 4, !tbaa !0
  %v57 = icmp slt i32 %v56, 11
  br i1 %v57, label %b16, label %b19

b16:                                              ; preds = %b15
  %v58 = getelementptr inbounds [0 x i32], [0 x i32]* @g11, i32 0, i32 %v56
  %v59 = load i32, i32* %v58, align 4, !tbaa !0
  %v60 = add nsw i32 %v59, 50
  %v61 = getelementptr inbounds [750 x i32], [750 x i32]* @g3, i32 0, i32 %v7
  %v62 = load i32, i32* %v61, align 4, !tbaa !0
  %v63 = getelementptr inbounds [450 x i32], [450 x i32]* @g0, i32 0, i32 %v62
  %v64 = load i32, i32* %v63, align 4, !tbaa !0
  %v65 = mul nsw i32 %v64, %v60
  %v66 = sdiv i32 %v65, 2
  %v67 = add nsw i32 %v66, %v6
  %v68 = getelementptr inbounds [750 x i32], [750 x i32]* @g8, i32 0, i32 %v7
  %v69 = load i32, i32* %v68, align 4, !tbaa !0
  %v70 = icmp sgt i32 %v69, 1
  br i1 %v70, label %b17, label %b18

b17:                                              ; preds = %b16
  %v71 = mul nsw i32 %v69, 25
  %v72 = add nsw i32 %v71, %v67
  br label %b18

b18:                                              ; preds = %b17, %b16
  %v73 = phi i32 [ %v72, %b17 ], [ %v67, %b16 ]
  %v74 = tail call i32 @f1(i32 %v7, i32 %a0)
  %v75 = add nsw i32 %v74, %v5
  br label %b19

b19:                                              ; preds = %b18, %b15, %b14, %b13, %b12, %b11, %b7, %b6, %b5, %b4, %b2
  %v76 = phi i32 [ %v6, %b7 ], [ %v6, %b6 ], [ %v6, %b5 ], [ %v6, %b4 ], [ %v73, %b18 ], [ %v6, %b15 ], [ %v6, %b14 ], [ %v6, %b13 ], [ %v6, %b12 ], [ %v6, %b11 ], [ %v6, %b2 ]
  %v77 = phi i32 [ %v5, %b7 ], [ %v5, %b6 ], [ %v5, %b5 ], [ %v5, %b4 ], [ %v75, %b18 ], [ %v5, %b15 ], [ %v5, %b14 ], [ %v5, %b13 ], [ %v5, %b12 ], [ %v5, %b11 ], [ %v5, %b2 ]
  %v78 = phi i32 [ %v4, %b7 ], [ %v4, %b6 ], [ %v4, %b5 ], [ %v4, %b4 ], [ %v4, %b18 ], [ %v4, %b15 ], [ %v4, %b14 ], [ %v4, %b13 ], [ %v44, %b12 ], [ %v4, %b11 ], [ %v4, %b2 ]
  %v79 = phi i32 [ %v29, %b7 ], [ %v3, %b6 ], [ %v3, %b5 ], [ %v3, %b4 ], [ %v3, %b18 ], [ %v3, %b15 ], [ %v3, %b14 ], [ %v3, %b13 ], [ %v3, %b12 ], [ %v3, %b11 ], [ %v3, %b2 ]
  %v80 = add nsw i32 %v7, 1
  %v81 = icmp slt i32 %v80, %v0
  br i1 %v81, label %b2, label %b20

b20:                                              ; preds = %b19
  br label %b21

b21:                                              ; preds = %b20, %b0
  %v82 = phi i32 [ 0, %b0 ], [ %v79, %b20 ]
  %v83 = phi i32 [ 32767, %b0 ], [ %v78, %b20 ]
  %v84 = phi i32 [ 0, %b0 ], [ %v77, %b20 ]
  %v85 = phi i32 [ 0, %b0 ], [ %v76, %b20 ]
  %v86 = icmp eq i32 %v83, 32767
  %v87 = sdiv i32 %v83, 2
  %v88 = select i1 %v86, i32 0, i32 %v87
  %v89 = add i32 %v84, %v85
  %v90 = add i32 %v89, %v82
  %v91 = add i32 %v90, %v88
  ret i32 %v91
}

; Function Attrs: nounwind readonly
declare i32 @f1(i32, i32) #0

attributes #0 = { nounwind readonly }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
