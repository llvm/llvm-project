; RUN: llc -march=hexagon -O3 < %s
; REQUIRES: asserts

target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
define void @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = alloca [8 x i32], align 8
  %v1 = bitcast [8 x i32]* %v0 to i8*
  call void @llvm.memset.p0i8.i32(i8* align 8 %v1, i8 0, i32 32, i1 false)
  %v2 = icmp sgt i32 %a0, 0
  br i1 %v2, label %b1, label %b18

b1:                                               ; preds = %b0
  %v3 = getelementptr inbounds [8 x i32], [8 x i32]* %v0, i32 0, i32 6
  %v4 = inttoptr i32 %a1 to i32*
  %v5 = add i32 %a0, -1
  %v6 = icmp sgt i32 %v5, 0
  br i1 %v6, label %b2, label %b13

b2:                                               ; preds = %b1
  %v7 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 0
  %v8 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 1
  %v9 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 2
  %v10 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 3
  %v11 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 4
  %v12 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 5
  %v13 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 6
  %v14 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 7
  %v15 = add i32 %a0, -2
  %v16 = lshr i32 %v15, 1
  %v17 = add i32 %v16, 1
  %v18 = urem i32 %v17, 2
  %v19 = icmp ne i32 %v18, 0
  %v20 = add i32 %v5, -2
  %v21 = icmp ugt i32 %v17, 1
  br i1 %v21, label %b3, label %b7

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b22, %b3
  %v22 = phi i32 [ 0, %b3 ], [ %v124, %b22 ]
  %v23 = phi i32 [ 0, %b3 ], [ %v136, %b22 ]
  %v24 = mul nsw i32 %v22, 4
  %v25 = add nsw i32 %v24, 268435456
  %v26 = inttoptr i32 %v25 to i32*
  store volatile i32 %a1, i32* %v26, align 4, !tbaa !0
  %v27 = load i32, i32* %v7, align 8, !tbaa !0
  store volatile i32 %v27, i32* %v4, align 4, !tbaa !0
  %v28 = load i32, i32* %v8, align 4, !tbaa !0
  store volatile i32 %v28, i32* %v4, align 4, !tbaa !0
  %v29 = load i32, i32* %v9, align 8, !tbaa !0
  store volatile i32 %v29, i32* %v4, align 4, !tbaa !0
  %v30 = load i32, i32* %v10, align 4, !tbaa !0
  store volatile i32 %v30, i32* %v4, align 4, !tbaa !0
  %v31 = load i32, i32* %v11, align 8, !tbaa !0
  store volatile i32 %v31, i32* %v4, align 4, !tbaa !0
  %v32 = load i32, i32* %v12, align 4, !tbaa !0
  store volatile i32 %v32, i32* %v4, align 4, !tbaa !0
  %v33 = load i32, i32* %v13, align 8, !tbaa !0
  store volatile i32 %v33, i32* %v4, align 4, !tbaa !0
  %v34 = load i32, i32* %v14, align 4, !tbaa !0
  store volatile i32 %v34, i32* %v4, align 4, !tbaa !0
  %v35 = icmp eq i32 %v23, 0
  br i1 %v35, label %b19, label %b20

b5:                                               ; preds = %b22
  %v36 = phi i32 [ %v136, %b22 ]
  %v37 = phi i32 [ %v124, %b22 ]
  br i1 %v19, label %b6, label %b12

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b6, %b2
  %v38 = phi i32 [ 0, %b2 ], [ %v36, %b6 ]
  %v39 = phi i32 [ 0, %b2 ], [ %v37, %b6 ]
  br label %b8

b8:                                               ; preds = %b10, %b7
  %v40 = phi i32 [ %v39, %b7 ], [ %v54, %b10 ]
  %v41 = phi i32 [ %v38, %b7 ], [ %v66, %b10 ]
  %v42 = mul nsw i32 %v40, 4
  %v43 = add nsw i32 %v42, 268435456
  %v44 = inttoptr i32 %v43 to i32*
  store volatile i32 %a1, i32* %v44, align 4, !tbaa !0
  %v45 = load i32, i32* %v7, align 8, !tbaa !0
  store volatile i32 %v45, i32* %v4, align 4, !tbaa !0
  %v46 = load i32, i32* %v8, align 4, !tbaa !0
  store volatile i32 %v46, i32* %v4, align 4, !tbaa !0
  %v47 = load i32, i32* %v9, align 8, !tbaa !0
  store volatile i32 %v47, i32* %v4, align 4, !tbaa !0
  %v48 = load i32, i32* %v10, align 4, !tbaa !0
  store volatile i32 %v48, i32* %v4, align 4, !tbaa !0
  %v49 = load i32, i32* %v11, align 8, !tbaa !0
  store volatile i32 %v49, i32* %v4, align 4, !tbaa !0
  %v50 = load i32, i32* %v12, align 4, !tbaa !0
  store volatile i32 %v50, i32* %v4, align 4, !tbaa !0
  %v51 = load i32, i32* %v13, align 8, !tbaa !0
  store volatile i32 %v51, i32* %v4, align 4, !tbaa !0
  %v52 = load i32, i32* %v14, align 4, !tbaa !0
  store volatile i32 %v52, i32* %v4, align 4, !tbaa !0
  %v53 = icmp eq i32 %v41, 0
  br i1 %v53, label %b9, label %b10

b9:                                               ; preds = %b8
  store i32 0, i32* %v3, align 8, !tbaa !0
  br label %b10

b10:                                              ; preds = %b9, %b8
  %v54 = phi i32 [ 3, %b9 ], [ %v40, %b8 ]
  %v55 = mul nsw i32 %v54, 4
  %v56 = add nsw i32 %v55, 268435456
  %v57 = inttoptr i32 %v56 to i32*
  store volatile i32 %a1, i32* %v57, align 4, !tbaa !0
  %v58 = load i32, i32* %v7, align 8, !tbaa !0
  store volatile i32 %v58, i32* %v4, align 4, !tbaa !0
  %v59 = load i32, i32* %v8, align 4, !tbaa !0
  store volatile i32 %v59, i32* %v4, align 4, !tbaa !0
  %v60 = load i32, i32* %v9, align 8, !tbaa !0
  store volatile i32 %v60, i32* %v4, align 4, !tbaa !0
  %v61 = load i32, i32* %v10, align 4, !tbaa !0
  store volatile i32 %v61, i32* %v4, align 4, !tbaa !0
  %v62 = load i32, i32* %v11, align 8, !tbaa !0
  store volatile i32 %v62, i32* %v4, align 4, !tbaa !0
  %v63 = load i32, i32* %v12, align 4, !tbaa !0
  store volatile i32 %v63, i32* %v4, align 4, !tbaa !0
  %v64 = load i32, i32* %v13, align 8, !tbaa !0
  store volatile i32 %v64, i32* %v4, align 4, !tbaa !0
  %v65 = load i32, i32* %v14, align 4, !tbaa !0
  store volatile i32 %v65, i32* %v4, align 4, !tbaa !0
  %v66 = add nsw i32 %v41, 2
  %v67 = icmp slt i32 %v66, %v5
  br i1 %v67, label %b8, label %b11

b11:                                              ; preds = %b10
  %v68 = phi i32 [ %v66, %b10 ]
  %v69 = phi i32 [ %v54, %b10 ]
  br label %b12

b12:                                              ; preds = %b11, %b5
  %v70 = phi i32 [ %v36, %b5 ], [ %v68, %b11 ]
  %v71 = phi i32 [ %v37, %b5 ], [ %v69, %b11 ]
  %v72 = icmp eq i32 %v70, %a0
  br i1 %v72, label %b18, label %b13

b13:                                              ; preds = %b12, %b1
  %v73 = phi i32 [ 0, %b1 ], [ %v70, %b12 ]
  %v74 = phi i32 [ 0, %b1 ], [ %v71, %b12 ]
  %v75 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 0
  %v76 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 1
  %v77 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 2
  %v78 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 3
  %v79 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 4
  %v80 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 5
  %v81 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 6
  %v82 = getelementptr [8 x i32], [8 x i32]* %v0, i32 0, i32 7
  br label %b14

b14:                                              ; preds = %b16, %b13
  %v83 = phi i32 [ %v74, %b13 ], [ %v86, %b16 ]
  %v84 = phi i32 [ %v73, %b13 ], [ %v98, %b16 ]
  %v85 = icmp eq i32 %v84, 1
  br i1 %v85, label %b15, label %b16

b15:                                              ; preds = %b14
  store i32 0, i32* %v3, align 8, !tbaa !0
  br label %b16

b16:                                              ; preds = %b15, %b14
  %v86 = phi i32 [ 3, %b15 ], [ %v83, %b14 ]
  %v87 = mul nsw i32 %v86, 4
  %v88 = add nsw i32 %v87, 268435456
  %v89 = inttoptr i32 %v88 to i32*
  store volatile i32 %a1, i32* %v89, align 4, !tbaa !0
  %v90 = load i32, i32* %v75, align 8, !tbaa !0
  store volatile i32 %v90, i32* %v4, align 4, !tbaa !0
  %v91 = load i32, i32* %v76, align 4, !tbaa !0
  store volatile i32 %v91, i32* %v4, align 4, !tbaa !0
  %v92 = load i32, i32* %v77, align 8, !tbaa !0
  store volatile i32 %v92, i32* %v4, align 4, !tbaa !0
  %v93 = load i32, i32* %v78, align 4, !tbaa !0
  store volatile i32 %v93, i32* %v4, align 4, !tbaa !0
  %v94 = load i32, i32* %v79, align 8, !tbaa !0
  store volatile i32 %v94, i32* %v4, align 4, !tbaa !0
  %v95 = load i32, i32* %v80, align 4, !tbaa !0
  store volatile i32 %v95, i32* %v4, align 4, !tbaa !0
  %v96 = load i32, i32* %v81, align 8, !tbaa !0
  store volatile i32 %v96, i32* %v4, align 4, !tbaa !0
  %v97 = load i32, i32* %v82, align 4, !tbaa !0
  store volatile i32 %v97, i32* %v4, align 4, !tbaa !0
  %v98 = add nsw i32 %v84, 1
  %v99 = icmp eq i32 %v98, %a0
  br i1 %v99, label %b17, label %b14

b17:                                              ; preds = %b16
  br label %b18

b18:                                              ; preds = %b17, %b12, %b0
  ret void

b19:                                              ; preds = %b4
  store i32 0, i32* %v3, align 8, !tbaa !0
  br label %b20

b20:                                              ; preds = %b19, %b4
  %v100 = phi i32 [ 3, %b19 ], [ %v22, %b4 ]
  %v101 = mul nsw i32 %v100, 4
  %v102 = add nsw i32 %v101, 268435456
  %v103 = inttoptr i32 %v102 to i32*
  store volatile i32 %a1, i32* %v103, align 4, !tbaa !0
  %v104 = load i32, i32* %v7, align 8, !tbaa !0
  store volatile i32 %v104, i32* %v4, align 4, !tbaa !0
  %v105 = load i32, i32* %v8, align 4, !tbaa !0
  store volatile i32 %v105, i32* %v4, align 4, !tbaa !0
  %v106 = load i32, i32* %v9, align 8, !tbaa !0
  store volatile i32 %v106, i32* %v4, align 4, !tbaa !0
  %v107 = load i32, i32* %v10, align 4, !tbaa !0
  store volatile i32 %v107, i32* %v4, align 4, !tbaa !0
  %v108 = load i32, i32* %v11, align 8, !tbaa !0
  store volatile i32 %v108, i32* %v4, align 4, !tbaa !0
  %v109 = load i32, i32* %v12, align 4, !tbaa !0
  store volatile i32 %v109, i32* %v4, align 4, !tbaa !0
  %v110 = load i32, i32* %v13, align 8, !tbaa !0
  store volatile i32 %v110, i32* %v4, align 4, !tbaa !0
  %v111 = load i32, i32* %v14, align 4, !tbaa !0
  store volatile i32 %v111, i32* %v4, align 4, !tbaa !0
  %v112 = add nsw i32 %v23, 2
  %v113 = mul nsw i32 %v100, 4
  %v114 = add nsw i32 %v113, 268435456
  %v115 = inttoptr i32 %v114 to i32*
  store volatile i32 %a1, i32* %v115, align 4, !tbaa !0
  %v116 = load i32, i32* %v7, align 8, !tbaa !0
  store volatile i32 %v116, i32* %v4, align 4, !tbaa !0
  %v117 = load i32, i32* %v8, align 4, !tbaa !0
  store volatile i32 %v117, i32* %v4, align 4, !tbaa !0
  %v118 = load i32, i32* %v9, align 8, !tbaa !0
  store volatile i32 %v118, i32* %v4, align 4, !tbaa !0
  %v119 = load i32, i32* %v10, align 4, !tbaa !0
  store volatile i32 %v119, i32* %v4, align 4, !tbaa !0
  %v120 = load i32, i32* %v11, align 8, !tbaa !0
  store volatile i32 %v120, i32* %v4, align 4, !tbaa !0
  %v121 = load i32, i32* %v12, align 4, !tbaa !0
  store volatile i32 %v121, i32* %v4, align 4, !tbaa !0
  %v122 = load i32, i32* %v13, align 8, !tbaa !0
  store volatile i32 %v122, i32* %v4, align 4, !tbaa !0
  %v123 = load i32, i32* %v14, align 4, !tbaa !0
  store volatile i32 %v123, i32* %v4, align 4, !tbaa !0
  br i1 false, label %b21, label %b22

b21:                                              ; preds = %b20
  store i32 0, i32* %v3, align 8, !tbaa !0
  br label %b22

b22:                                              ; preds = %b21, %b20
  %v124 = phi i32 [ 3, %b21 ], [ %v100, %b20 ]
  %v125 = mul nsw i32 %v124, 4
  %v126 = add nsw i32 %v125, 268435456
  %v127 = inttoptr i32 %v126 to i32*
  store volatile i32 %a1, i32* %v127, align 4, !tbaa !0
  %v128 = load i32, i32* %v7, align 8, !tbaa !0
  store volatile i32 %v128, i32* %v4, align 4, !tbaa !0
  %v129 = load i32, i32* %v8, align 4, !tbaa !0
  store volatile i32 %v129, i32* %v4, align 4, !tbaa !0
  %v130 = load i32, i32* %v9, align 8, !tbaa !0
  store volatile i32 %v130, i32* %v4, align 4, !tbaa !0
  %v131 = load i32, i32* %v10, align 4, !tbaa !0
  store volatile i32 %v131, i32* %v4, align 4, !tbaa !0
  %v132 = load i32, i32* %v11, align 8, !tbaa !0
  store volatile i32 %v132, i32* %v4, align 4, !tbaa !0
  %v133 = load i32, i32* %v12, align 4, !tbaa !0
  store volatile i32 %v133, i32* %v4, align 4, !tbaa !0
  %v134 = load i32, i32* %v13, align 8, !tbaa !0
  store volatile i32 %v134, i32* %v4, align 4, !tbaa !0
  %v135 = load i32, i32* %v14, align 4, !tbaa !0
  store volatile i32 %v135, i32* %v4, align 4, !tbaa !0
  %v136 = add nsw i32 %v112, 2
  %v137 = icmp slt i32 %v136, %v20
  br i1 %v137, label %b4, label %b5
}

; Function Attrs: nounwind
define void @f1(i32 %a0, i32 %a1) #0 {
b0:
  tail call void @f0(i32 %a0, i32 %a1)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
