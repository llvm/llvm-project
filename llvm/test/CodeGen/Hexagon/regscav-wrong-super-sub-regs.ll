; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; Test that register scvenging does not assert because of wrong
; bits being set for Kill and Def bit vectors in replaceSuperBySubRegs

%s.0 = type { i32, ptr, [0 x i32], [0 x i32], [1 x i32] }
%s.1 = type { %s.2, %s.4, %s.5 }
%s.2 = type { %s.3 }
%s.3 = type { i32 }
%s.4 = type { i32 }
%s.5 = type { [0 x i32], [0 x ptr] }

@g0 = common global i32 0, align 4
@g1 = common global %s.0 zeroinitializer, align 4
@g2 = common global i32 0, align 4
@g3 = common global i32 0, align 4
@g4 = common global ptr null, align 4
@g5 = common global i32 0, align 4
@g6 = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @f0(ptr nocapture readonly %a0) #0 {
b0:
  %v0 = alloca [0 x i32], align 4
  %v1 = load i32, ptr @g0, align 4, !tbaa !0
  %v3 = load i32, ptr %a0, align 4, !tbaa !0
  %v4 = load ptr, ptr getelementptr inbounds (%s.0, ptr @g1, i32 0, i32 1), align 4, !tbaa !4
  %v5 = load i32, ptr @g2, align 4, !tbaa !0
  %v6 = sub i32 0, %v5
  %v7 = getelementptr inbounds i32, ptr %v4, i32 %v6
  %v8 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 1, i32 0
  %v9 = load i32, ptr %v8, align 4, !tbaa !0
  switch i32 %v9, label %b17 [
    i32 0, label %b1
    i32 1, label %b2
  ]

b1:                                               ; preds = %b0
  store i32 0, ptr @g3, align 4, !tbaa !0
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v10 = icmp eq i32 %v1, 0
  %v11 = icmp sgt i32 %v3, 0
  %v13 = sdiv i32 %v3, 2
  %v14 = add i32 %v13, -1
  %v15 = getelementptr inbounds [0 x i32], ptr %v0, i32 0, i32 1
  %v16 = getelementptr inbounds [0 x i32], ptr %v0, i32 0, i32 2
  %v17 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 2, i32 1, i32 %v1
  %v18 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 2, i32 1, i32 0
  %v19 = sub i32 1, %v5
  %v20 = getelementptr inbounds i32, ptr %v4, i32 %v19
  %v21 = sdiv i32 %v3, 4
  %v22 = icmp slt i32 %v3, -3
  %v23 = add i32 %v3, -1
  %v24 = lshr i32 %v23, 2
  %v25 = mul i32 %v24, 4
  %v26 = add i32 %v25, 4
  %v27 = add i32 %v13, -2
  %v28 = icmp slt i32 %v26, 0
  %v29 = add i32 %v21, 1
  %v30 = select i1 %v22, i32 1, i32 %v29
  br label %b4

b3:                                               ; preds = %b16
  store i32 %v30, ptr @g3, align 4, !tbaa !0
  br label %b4

b4:                                               ; preds = %b13, %b3, %b2
  %v31 = phi i32 [ undef, %b2 ], [ %v87, %b3 ], [ %v87, %b13 ]
  %v32 = phi i32 [ undef, %b2 ], [ %v86, %b3 ], [ %v86, %b13 ]
  %v33 = phi i32 [ undef, %b2 ], [ %v35, %b3 ], [ %v35, %b13 ]
  %v34 = phi i32 [ undef, %b2 ], [ %v89, %b3 ], [ %v89, %b13 ]
  %v35 = phi i32 [ undef, %b2 ], [ %v94, %b3 ], [ %v65, %b13 ]
  br i1 %v10, label %b6, label %b5

b5:                                               ; preds = %b5, %b4
  br label %b5

b6:                                               ; preds = %b4
  br i1 %v11, label %b8, label %b7

b7:                                               ; preds = %b6
  store i32 0, ptr @g3, align 4, !tbaa !0
  br label %b11

b8:                                               ; preds = %b6
  store i32 %v26, ptr @g3, align 4, !tbaa !0
  br i1 %v28, label %b9, label %b11

b9:                                               ; preds = %b8
  %v36 = load ptr, ptr @g4, align 4, !tbaa !7
  br label %b10

b10:                                              ; preds = %b10, %b9
  %v37 = phi i32 [ %v26, %b9 ], [ %v45, %b10 ]
  %v38 = phi i32 [ %v34, %b9 ], [ %v44, %b10 ]
  %v39 = add nsw i32 %v37, %v33
  %v40 = shl i32 %v39, 1
  %v41 = getelementptr inbounds i32, ptr %v36, i32 %v40
  %v42 = load i32, ptr %v41, align 4, !tbaa !0
  %v43 = icmp slt i32 %v42, %v31
  %v44 = select i1 %v43, i32 0, i32 %v38
  %v45 = add nsw i32 %v37, 1
  store i32 %v45, ptr @g3, align 4, !tbaa !0
  %v46 = icmp slt i32 %v45, 0
  br i1 %v46, label %b10, label %b11

b11:                                              ; preds = %b10, %b8, %b7
  %v47 = phi i32 [ %v26, %b8 ], [ 0, %b7 ], [ 0, %b10 ]
  %v48 = phi i32 [ %v34, %b8 ], [ %v34, %b7 ], [ %v44, %b10 ]
  %v49 = load i32, ptr @g5, align 4, !tbaa !0
  %v50 = icmp slt i32 %v13, %v49
  %v51 = icmp slt i32 %v47, %v14
  %v52 = and i1 %v50, %v51
  br i1 %v52, label %b12, label %b13

b12:                                              ; preds = %b11
  %v53 = sub i32 %v27, %v47
  %v54 = lshr i32 %v53, 1
  %v55 = mul i32 %v54, 2
  %v56 = add i32 %v47, 2
  %v57 = add i32 %v56, %v55
  store i32 %v57, ptr @g3, align 4, !tbaa !0
  br label %b13

b13:                                              ; preds = %b12, %b11
  %v58 = shl i32 %v35, 2
  %v59 = load ptr, ptr @g4, align 4, !tbaa !7
  %v60 = getelementptr inbounds i32, ptr %v59, i32 %v58
  %v61 = load i32, ptr %v60, align 4, !tbaa !0
  %v62 = load i32, ptr %v7, align 4, !tbaa !0
  %v63 = add nsw i32 %v62, %v61
  %v64 = add nsw i32 %v63, %v32
  store i32 %v64, ptr %v15, align 4, !tbaa !0
  %v65 = add i32 %v35, -1
  %v66 = getelementptr inbounds i32, ptr %v59, i32 %v65
  %v67 = load i32, ptr %v66, align 4, !tbaa !0
  %v68 = sub i32 %v49, %v5
  %v69 = getelementptr inbounds i32, ptr %v4, i32 %v68
  %v70 = load i32, ptr %v69, align 4, !tbaa !0
  %v71 = add nsw i32 %v70, %v67
  %v72 = load i32, ptr %v16, align 4, !tbaa !0
  %v73 = add nsw i32 %v71, %v72
  store i32 %v73, ptr %v16, align 4, !tbaa !0
  %v74 = load i32, ptr @g6, align 4, !tbaa !0
  %v75 = load ptr, ptr %v17, align 4, !tbaa !7
  %v76 = load i32, ptr getelementptr inbounds (%s.0, ptr @g1, i32 0, i32 4, i32 0), align 4, !tbaa !0
  %v77 = call i32 %v75(ptr getelementptr inbounds (%s.0, ptr @g1, i32 0, i32 4, i32 0), ptr null, ptr null, ptr null, ptr null, i32 %v76, ptr null) #0
  %v78 = load ptr, ptr %v18, align 4, !tbaa !7
  %v79 = inttoptr i32 %v74 to ptr
  %v80 = load i32, ptr getelementptr inbounds (%s.0, ptr @g1, i32 0, i32 4, i32 0), align 4, !tbaa !0
  %v81 = call i32 %v78(ptr getelementptr inbounds (%s.0, ptr @g1, i32 0, i32 4, i32 0), ptr null, ptr null, ptr null, ptr %v79, i32 %v80, ptr %v0) #0
  %v82 = load ptr, ptr @g4, align 4, !tbaa !7
  %v83 = getelementptr inbounds i32, ptr %v82, i32 %v58
  %v84 = load i32, ptr %v83, align 4, !tbaa !0
  %v85 = load i32, ptr %v20, align 4, !tbaa !0
  %v86 = add nsw i32 %v85, %v84
  store i32 %v86, ptr %v15, align 4, !tbaa !0
  %v87 = load i32, ptr %v0, align 4, !tbaa !0
  %v88 = icmp eq i32 %v87, 0
  %v89 = select i1 %v88, i32 %v48, i32 1
  store i32 %v89, ptr @g5, align 4, !tbaa !0
  store i32 0, ptr @g3, align 4, !tbaa !0
  br i1 %v22, label %b4, label %b14

b14:                                              ; preds = %b16, %b13
  %v90 = phi i32 [ %v95, %b16 ], [ 0, %b13 ]
  %v91 = phi i32 [ %v94, %b16 ], [ %v65, %b13 ]
  br i1 %v88, label %b16, label %b15

b15:                                              ; preds = %b14
  %v92 = mul i32 %v90, -4
  %v93 = add nsw i32 %v92, 1
  br label %b16

b16:                                              ; preds = %b15, %b14
  %v94 = phi i32 [ %v93, %b15 ], [ %v91, %b14 ]
  %v95 = add nsw i32 %v90, 1
  %v96 = icmp slt i32 %v90, %v21
  br i1 %v96, label %b14, label %b3

b17:                                              ; preds = %b0
  ret i32 undef
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !6, i64 4}
!5 = !{!"", !1, i64 0, !6, i64 4, !2, i64 8, !2, i64 8, !2, i64 8}
!6 = !{!"any pointer", !2, i64 0}
!7 = !{!6, !6, i64 0}
