; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: loop0(.LBB0_{{[0-9]+}},#347)

target triple = "hexagon"

; Function Attrs: norecurse nounwind optsize readonly
define i32 @f0(i32 %a0, i8* nocapture readonly %a1, i32 %a2) local_unnamed_addr #0 {
b0:
  %v0 = lshr i32 %a0, 16
  %v1 = and i32 %a0, 65535
  %v2 = icmp ugt i32 %a2, 5551
  br i1 %v2, label %b1, label %b4

b1:                                               ; preds = %b0, %b3
  %v3 = phi i32 [ %v96, %b3 ], [ %v0, %b0 ]
  %v4 = phi i32 [ %v7, %b3 ], [ %a2, %b0 ]
  %v5 = phi i8* [ %v94, %b3 ], [ %a1, %b0 ]
  %v6 = phi i32 [ %v95, %b3 ], [ %v1, %b0 ]
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v8 = phi i32 [ %v6, %b1 ], [ %v89, %b2 ]
  %v9 = phi i8* [ %v5, %b1 ], [ %v91, %b2 ]
  %v10 = phi i32 [ %v3, %b1 ], [ %v90, %b2 ]
  %v11 = phi i32 [ 347, %b1 ], [ %v92, %b2 ]
  %v12 = load i8, i8* %v9, align 1, !tbaa !0
  %v13 = zext i8 %v12 to i32
  %v14 = add i32 %v8, %v13
  %v15 = add i32 %v14, %v10
  %v16 = getelementptr inbounds i8, i8* %v9, i32 1
  %v17 = load i8, i8* %v16, align 1, !tbaa !0
  %v18 = zext i8 %v17 to i32
  %v19 = add i32 %v14, %v18
  %v20 = add i32 %v15, %v19
  %v21 = getelementptr inbounds i8, i8* %v9, i32 2
  %v22 = load i8, i8* %v21, align 1, !tbaa !0
  %v23 = zext i8 %v22 to i32
  %v24 = add i32 %v19, %v23
  %v25 = add i32 %v20, %v24
  %v26 = getelementptr inbounds i8, i8* %v9, i32 3
  %v27 = load i8, i8* %v26, align 1, !tbaa !0
  %v28 = zext i8 %v27 to i32
  %v29 = add i32 %v24, %v28
  %v30 = add i32 %v25, %v29
  %v31 = getelementptr inbounds i8, i8* %v9, i32 4
  %v32 = load i8, i8* %v31, align 1, !tbaa !0
  %v33 = zext i8 %v32 to i32
  %v34 = add i32 %v29, %v33
  %v35 = add i32 %v30, %v34
  %v36 = getelementptr inbounds i8, i8* %v9, i32 5
  %v37 = load i8, i8* %v36, align 1, !tbaa !0
  %v38 = zext i8 %v37 to i32
  %v39 = add i32 %v34, %v38
  %v40 = add i32 %v35, %v39
  %v41 = getelementptr inbounds i8, i8* %v9, i32 6
  %v42 = load i8, i8* %v41, align 1, !tbaa !0
  %v43 = zext i8 %v42 to i32
  %v44 = add i32 %v39, %v43
  %v45 = add i32 %v40, %v44
  %v46 = getelementptr inbounds i8, i8* %v9, i32 7
  %v47 = load i8, i8* %v46, align 1, !tbaa !0
  %v48 = zext i8 %v47 to i32
  %v49 = add i32 %v44, %v48
  %v50 = add i32 %v45, %v49
  %v51 = getelementptr inbounds i8, i8* %v9, i32 8
  %v52 = load i8, i8* %v51, align 1, !tbaa !0
  %v53 = zext i8 %v52 to i32
  %v54 = add i32 %v49, %v53
  %v55 = add i32 %v50, %v54
  %v56 = getelementptr inbounds i8, i8* %v9, i32 9
  %v57 = load i8, i8* %v56, align 1, !tbaa !0
  %v58 = zext i8 %v57 to i32
  %v59 = add i32 %v54, %v58
  %v60 = add i32 %v55, %v59
  %v61 = getelementptr inbounds i8, i8* %v9, i32 10
  %v62 = load i8, i8* %v61, align 1, !tbaa !0
  %v63 = zext i8 %v62 to i32
  %v64 = add i32 %v59, %v63
  %v65 = add i32 %v60, %v64
  %v66 = getelementptr inbounds i8, i8* %v9, i32 11
  %v67 = load i8, i8* %v66, align 1, !tbaa !0
  %v68 = zext i8 %v67 to i32
  %v69 = add i32 %v64, %v68
  %v70 = add i32 %v65, %v69
  %v71 = getelementptr inbounds i8, i8* %v9, i32 12
  %v72 = load i8, i8* %v71, align 1, !tbaa !0
  %v73 = zext i8 %v72 to i32
  %v74 = add i32 %v69, %v73
  %v75 = add i32 %v70, %v74
  %v76 = getelementptr inbounds i8, i8* %v9, i32 13
  %v77 = load i8, i8* %v76, align 1, !tbaa !0
  %v78 = zext i8 %v77 to i32
  %v79 = add i32 %v74, %v78
  %v80 = add i32 %v75, %v79
  %v81 = getelementptr inbounds i8, i8* %v9, i32 14
  %v82 = load i8, i8* %v81, align 1, !tbaa !0
  %v83 = zext i8 %v82 to i32
  %v84 = add i32 %v79, %v83
  %v85 = add i32 %v80, %v84
  %v86 = getelementptr inbounds i8, i8* %v9, i32 15
  %v87 = load i8, i8* %v86, align 1, !tbaa !0
  %v88 = zext i8 %v87 to i32
  %v89 = add i32 %v84, %v88
  %v90 = add i32 %v85, %v89
  %v91 = getelementptr inbounds i8, i8* %v9, i32 16
  %v92 = add nsw i32 %v11, -1
  %v93 = icmp eq i32 %v92, 0
  br i1 %v93, label %b3, label %b2

b3:                                               ; preds = %b2
  %v7 = add i32 %v4, -5552
  %v94 = getelementptr i8, i8* %v5, i32 5552
  %v95 = urem i32 %v89, 65521
  %v96 = urem i32 %v90, 65521
  %v97 = icmp ugt i32 %v7, 5551
  br i1 %v97, label %b1, label %b4

b4:                                               ; preds = %b3, %b0
  %v98 = phi i32 [ %v0, %b0 ], [ %v96, %b3 ]
  %v99 = phi i32 [ %v1, %b0 ], [ %v95, %b3 ]
  %v100 = shl nuw i32 %v98, 16
  %v101 = or i32 %v100, %v99
  ret i32 %v101
}

attributes #0 = { norecurse nounwind optsize readonly "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
