; RUN: llc -disable-lsr -march=hexagon -enable-aa-sched-mi -O2 < %s
; REQUIRES: asserts

; Test when there is a Phi operand that is defined by another Phi, but
; the two Phis are scheduled in different iterations.

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, i32 %a1, i32 %a2, i32 %a3, ptr noalias nocapture %a4, i32 %a5) #0 {
b0:
  %v0 = add i32 %a2, -1
  %v1 = icmp ugt i32 %v0, 1
  br i1 %v1, label %b1, label %b6

b1:                                               ; preds = %b0
  %v2 = add i32 %a1, -1
  %v3 = mul i32 %a3, 2
  %v4 = add i32 %v3, 1
  %v5 = add i32 %a3, 1
  %v6 = add i32 %a1, -2
  %v7 = getelementptr i8, ptr %a0, i32 2
  %v8 = add i32 %a5, 1
  %v9 = getelementptr i8, ptr %a4, i32 %v8
  br label %b2

b2:                                               ; preds = %b5, %b1
  %v10 = phi ptr [ %v85, %b5 ], [ %v9, %b1 ]
  %v11 = phi ptr [ %v84, %b5 ], [ %v7, %b1 ]
  %v12 = phi i32 [ 0, %b1 ], [ %v83, %b5 ]
  %v13 = phi i32 [ 1, %b1 ], [ %v82, %b5 ]
  %v14 = icmp ugt i32 %v2, 1
  %v15 = mul i32 %v12, %a3
  br i1 %v14, label %b3, label %b5

b3:                                               ; preds = %b2
  %v16 = add i32 %v12, 2
  %v17 = add i32 %v15, 1
  %v18 = mul i32 %v16, %a3
  %v19 = add i32 %v4, %v15
  %v20 = add i32 %v15, %a3
  %v21 = add i32 %v5, %v15
  %v22 = getelementptr i8, ptr %a0, i32 %v15
  %v23 = getelementptr i8, ptr %a0, i32 %v17
  %v24 = getelementptr i8, ptr %a0, i32 %v18
  %v25 = getelementptr i8, ptr %a0, i32 %v19
  %v26 = getelementptr i8, ptr %a0, i32 %v20
  %v27 = getelementptr i8, ptr %a0, i32 %v21
  %v28 = load i8, ptr %v23, align 1
  %v29 = load i8, ptr %v22, align 1
  %v30 = load i8, ptr %v25, align 1
  %v31 = load i8, ptr %v24, align 1
  %v32 = load i8, ptr %v27, align 1
  %v33 = load i8, ptr %v26, align 1
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v34 = phi ptr [ %v80, %b4 ], [ %v10, %b3 ]
  %v35 = phi ptr [ %v79, %b4 ], [ %v11, %b3 ]
  %v36 = phi i32 [ %v78, %b4 ], [ %v6, %b3 ]
  %v37 = phi i8 [ %v28, %b3 ], [ %v43, %b4 ]
  %v38 = phi i8 [ %v29, %b3 ], [ %v37, %b4 ]
  %v39 = phi i8 [ %v30, %b3 ], [ %v47, %b4 ]
  %v40 = phi i8 [ %v31, %b3 ], [ %v39, %b4 ]
  %v41 = phi i8 [ %v32, %b3 ], [ %v45, %b4 ]
  %v42 = phi i8 [ %v33, %b3 ], [ %v41, %b4 ]
  %v43 = load i8, ptr %v35, align 1, !tbaa !0
  %v44 = getelementptr i8, ptr %v35, i32 %a3
  %v45 = load i8, ptr %v44, align 1, !tbaa !0
  %v46 = getelementptr i8, ptr %v35, i32 %v3
  %v47 = load i8, ptr %v46, align 1, !tbaa !0
  %v48 = zext i8 %v38 to i32
  %v49 = zext i8 %v37 to i32
  %v50 = zext i8 %v43 to i32
  %v51 = zext i8 %v40 to i32
  %v52 = zext i8 %v39 to i32
  %v53 = zext i8 %v47 to i32
  %v54 = sub i32 %v49, %v52
  %v55 = mul i32 %v54, 2
  %v56 = add i32 %v50, %v48
  %v57 = sub i32 %v56, %v51
  %v58 = sub i32 %v57, %v53
  %v59 = add i32 %v58, %v55
  %v60 = zext i8 %v42 to i32
  %v61 = zext i8 %v45 to i32
  %v62 = sub i32 %v60, %v61
  %v63 = mul i32 %v62, 2
  %v64 = sub i32 %v48, %v50
  %v65 = add i32 %v64, %v51
  %v66 = add i32 %v65, %v63
  %v67 = sub i32 %v66, %v53
  %v68 = icmp sgt i32 %v59, -1
  %v69 = sub i32 0, %v59
  %v70 = select i1 %v68, i32 %v59, i32 %v69
  %v71 = icmp sgt i32 %v67, -1
  %v72 = sub i32 0, %v67
  %v73 = select i1 %v71, i32 %v67, i32 %v72
  %v74 = add nsw i32 %v70, %v73
  %v75 = icmp ugt i32 %v74, 255
  %v76 = trunc i32 %v74 to i8
  %v77 = select i1 %v75, i8 -1, i8 %v76
  store i8 %v77, ptr %v34, align 1, !tbaa !0
  %v78 = add i32 %v36, -1
  %v79 = getelementptr i8, ptr %v35, i32 1
  %v80 = getelementptr i8, ptr %v34, i32 1
  %v81 = icmp eq i32 %v78, 0
  br i1 %v81, label %b5, label %b4

b5:                                               ; preds = %b4, %b2
  %v82 = add i32 %v13, 1
  %v83 = add i32 %v12, 1
  %v84 = getelementptr i8, ptr %v11, i32 %a3
  %v85 = getelementptr i8, ptr %v10, i32 %a5
  %v86 = icmp eq i32 %v82, %v0
  br i1 %v86, label %b6, label %b2

b6:                                               ; preds = %b5, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
