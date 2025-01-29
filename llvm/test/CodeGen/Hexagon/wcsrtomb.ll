; RUN: llc -mtriple=hexagon -O2 < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { i32, i16, i16 }

; Function Attrs: nounwind
define i32 @f0(ptr %a0, ptr nocapture %a1, i32 %a2, ptr %a3) #0 {
b0:
  %v0 = alloca [8 x i8], align 8
  %v1 = load ptr, ptr %a1, align 4, !tbaa !0
  %v2 = icmp eq ptr %a3, null
  br i1 %v2, label %b1, label %b2

b1:                                               ; preds = %b0
  %v3 = call ptr @f1() #1
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v4 = phi ptr [ %v3, %b1 ], [ %a3, %b0 ]
  %v5 = icmp eq ptr %a0, null
  br i1 %v5, label %b5, label %b3

b3:                                               ; preds = %b2
  %v6 = icmp eq i32 %a2, 0
  br i1 %v6, label %b23, label %b4

b4:                                               ; preds = %b3
  %v9 = getelementptr inbounds %s.0, ptr %v4, i32 0, i32 1
  %v10 = getelementptr inbounds %s.0, ptr %v4, i32 0, i32 2
  br label %b11

b5:                                               ; preds = %b2
  %v13 = load i32, ptr %v1, align 4, !tbaa !4
  %v14 = call i32 @f2(ptr %v0, i32 %v13, ptr %v4) #1
  %v15 = icmp slt i32 %v14, 0
  br i1 %v15, label %b25, label %b6

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b10, %b6
  %v16 = phi i32 [ %v29, %b10 ], [ %v14, %b6 ]
  %v17 = phi i32 [ %v26, %b10 ], [ 0, %b6 ]
  %v18 = phi ptr [ %v27, %b10 ], [ %v1, %b6 ]
  %v19 = icmp sgt i32 %v16, 0
  br i1 %v19, label %b8, label %b10

b8:                                               ; preds = %b7
  %v20 = add nsw i32 %v16, -1
  %v21 = getelementptr inbounds [8 x i8], ptr %v0, i32 0, i32 %v20
  %v22 = load i8, ptr %v21, align 1, !tbaa !6
  %v23 = icmp eq i8 %v22, 0
  br i1 %v23, label %b9, label %b10

b9:                                               ; preds = %b8
  %v24 = add i32 %v17, -1
  %v25 = add i32 %v24, %v16
  br label %b25

b10:                                              ; preds = %b8, %b7
  %v26 = add i32 %v16, %v17
  %v27 = getelementptr inbounds i32, ptr %v18, i32 1
  %v28 = load i32, ptr %v27, align 4, !tbaa !4
  %v29 = call i32 @f2(ptr %v0, i32 %v28, ptr %v4) #1
  %v30 = icmp slt i32 %v29, 0
  br i1 %v30, label %b24, label %b7

b11:                                              ; preds = %b21, %b4
  %v31 = phi ptr [ %a0, %b4 ], [ %v64, %b21 ]
  %v32 = phi i32 [ %a2, %b4 ], [ %v65, %b21 ]
  %v33 = phi i32 [ 0, %b4 ], [ %v62, %b21 ]
  %v34 = phi ptr [ %v1, %b4 ], [ %v63, %b21 ]
  %v35 = phi i32 [ undef, %b4 ], [ %v47, %b21 ]
  %v36 = phi i16 [ undef, %b4 ], [ %v46, %b21 ]
  %v37 = phi i16 [ undef, %b4 ], [ %v45, %b21 ]
  %v38 = call i32 @f3() #1
  %v39 = icmp ult i32 %v32, %v38
  br i1 %v39, label %b12, label %b13

b12:                                              ; preds = %b11
  %v40 = load i32, ptr %v4, align 4
  %v41 = load i32, ptr %v9, align 4
  %v42 = trunc i32 %v41 to i16
  %v43 = lshr i32 %v41, 16
  %v44 = trunc i32 %v43 to i16
  br label %b13

b13:                                              ; preds = %b12, %b11
  %v45 = phi i16 [ %v44, %b12 ], [ %v37, %b11 ]
  %v46 = phi i16 [ %v42, %b12 ], [ %v36, %b11 ]
  %v47 = phi i32 [ %v40, %b12 ], [ %v35, %b11 ]
  %v48 = phi ptr [ %v0, %b12 ], [ %v31, %b11 ]
  %v49 = load i32, ptr %v34, align 4, !tbaa !4
  %v50 = call i32 @f2(ptr %v48, i32 %v49, ptr %v4) #1
  %v51 = icmp slt i32 %v50, 0
  br i1 %v51, label %b22, label %b14

b14:                                              ; preds = %b13
  %v52 = icmp eq ptr %v31, %v48
  br i1 %v52, label %b18, label %b15

b15:                                              ; preds = %b14
  %v53 = icmp ult i32 %v32, %v50
  br i1 %v53, label %b16, label %b17

b16:                                              ; preds = %b15
  store i32 %v47, ptr %v4, align 4
  store i16 %v46, ptr %v9, align 4
  store i16 %v45, ptr %v10, align 2
  br label %b23

b17:                                              ; preds = %b15
  %v54 = call ptr @f4(ptr %v31, ptr %v0, i32 %v50) #1
  br label %b18

b18:                                              ; preds = %b17, %b14
  %v55 = icmp sgt i32 %v50, 0
  br i1 %v55, label %b19, label %b21

b19:                                              ; preds = %b18
  %v56 = add nsw i32 %v50, -1
  %v57 = getelementptr inbounds i8, ptr %v31, i32 %v56
  %v58 = load i8, ptr %v57, align 1, !tbaa !6
  %v59 = icmp eq i8 %v58, 0
  br i1 %v59, label %b20, label %b21

b20:                                              ; preds = %b19
  store ptr null, ptr %a1, align 4, !tbaa !0
  %v60 = add i32 %v33, -1
  %v61 = add i32 %v60, %v50
  br label %b25

b21:                                              ; preds = %b19, %b18
  %v62 = add i32 %v50, %v33
  %v63 = getelementptr inbounds i32, ptr %v34, i32 1
  %v64 = getelementptr inbounds i8, ptr %v31, i32 %v50
  %v65 = sub i32 %v32, %v50
  %v66 = icmp eq i32 %v32, %v50
  br i1 %v66, label %b22, label %b11

b22:                                              ; preds = %b21, %b13
  %v67 = phi ptr [ %v34, %b13 ], [ %v63, %b21 ]
  %v68 = phi i32 [ -1, %b13 ], [ %v62, %b21 ]
  br label %b23

b23:                                              ; preds = %b22, %b16, %b3
  %v69 = phi ptr [ %v34, %b16 ], [ %v1, %b3 ], [ %v67, %b22 ]
  %v70 = phi i32 [ %v33, %b16 ], [ 0, %b3 ], [ %v68, %b22 ]
  store ptr %v69, ptr %a1, align 4, !tbaa !0
  br label %b25

b24:                                              ; preds = %b10
  br label %b25

b25:                                              ; preds = %b24, %b23, %b20, %b9, %b5
  %v71 = phi i32 [ %v25, %b9 ], [ %v70, %b23 ], [ %v61, %b20 ], [ -1, %b5 ], [ -1, %b24 ]
  ret i32 %v71
}

declare ptr @f1(...)

declare i32 @f2(ptr, i32, ptr)

declare i32 @f3()

declare ptr @f4(ptr, ptr, i32)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2}
!6 = !{!2, !2, i64 0}
