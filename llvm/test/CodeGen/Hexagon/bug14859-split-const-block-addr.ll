; RUN: llc -march=hexagon -O3 -hexagon-small-data-threshold=0 < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { %s.1, %s.1* }
%s.1 = type { i8*, i8*, i8*, i32 }

; Function Attrs: nounwind
declare i32 @f0(%s.0* nocapture) #0 align 2

; Function Attrs: nounwind
declare void @f1(%s.0* nocapture) unnamed_addr #0 align 2

; Function Attrs: inlinehint
define void @f2(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, %s.0* %a5, i1 (i8, i8)* %a6) #1 {
b0:
  %v0 = alloca %s.0, align 4
  %v1 = alloca %s.0, align 4
  %v2 = alloca %s.0, align 4
  %v3 = alloca %s.0, align 4
  %v4 = alloca %s.0, align 4
  %v5 = alloca %s.0, align 4
  %v6 = inttoptr i32 %a0 to i8*
  %v7 = inttoptr i32 %a1 to i8*
  %v8 = add nsw i32 %a4, %a3
  %v9 = icmp eq i32 %v8, 2
  br i1 %v9, label %b1, label %b2

b1:                                               ; preds = %b0
  call void @f7(i8* %v7, i8* %v6, i1 (i8, i8)* %a6)
  br label %b43

b2:                                               ; preds = %b0
  %v10 = icmp sgt i32 %a3, %a4
  br i1 %v10, label %b18, label %b3

b3:                                               ; preds = %b2
  %v11 = call i32 @f0(%s.0* %a5)
  %v12 = icmp slt i32 %v11, %a3
  br i1 %v12, label %b18, label %b4

b4:                                               ; preds = %b3
  %v13 = getelementptr inbounds %s.0, %s.0* %a5, i32 0, i32 1
  %v14 = load %s.1*, %s.1** %v13, align 4, !tbaa !0
  %v15 = getelementptr inbounds %s.1, %s.1* %v14, i32 0, i32 0
  %v16 = load i8*, i8** %v15, align 4, !tbaa !0
  %v17 = getelementptr inbounds %s.1, %s.1* %v14, i32 0, i32 1
  store i8* %v16, i8** %v17, align 4, !tbaa !0
  %v18 = bitcast %s.0* %v3 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %v18, i8 0, i64 16, i1 false)
  %v19 = load %s.1*, %s.1** %v13, align 4, !tbaa !0
  %v20 = getelementptr inbounds %s.0, %s.0* %v3, i32 0, i32 1
  store %s.1* %v19, %s.1** %v20, align 4, !tbaa !0
  %v21 = bitcast %s.0* %v1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %v21)
  call void @llvm.memset.p0i8.i64(i8* align 4 %v21, i8 0, i64 16, i1 false)
  %v22 = getelementptr inbounds %s.0, %s.0* %v1, i32 0, i32 1
  store %s.1* %v19, %s.1** %v22, align 4, !tbaa !0
  %v23 = icmp eq i8* %v6, %v7
  br i1 %v23, label %b6, label %b5

b5:                                               ; preds = %b4
  call void @f8(i8* %v6, %s.0* %v1, i8* %v7)
  %v24 = load %s.1*, %s.1** %v22, align 4, !tbaa !0
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v25 = phi %s.1* [ %v24, %b5 ], [ %v19, %b4 ]
  %v26 = bitcast %s.0* %v2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %v26, i8 0, i64 16, i1 false)
  %v27 = getelementptr inbounds %s.0, %s.0* %v2, i32 0, i32 1
  store %s.1* %v25, %s.1** %v27, align 4, !tbaa !0
  call void @f1(%s.0* %v1) #0
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %v21)
  call void @f1(%s.0* %v2) #0
  call void @f1(%s.0* %v3) #0
  %v28 = load %s.1*, %s.1** %v13, align 4, !tbaa !0
  %v29 = getelementptr inbounds %s.1, %s.1* %v28, i32 0, i32 0
  %v30 = load i8*, i8** %v29, align 4, !tbaa !0
  %v31 = getelementptr inbounds %s.1, %s.1* %v28, i32 0, i32 1
  %v32 = load i8*, i8** %v31, align 4, !tbaa !0
  %v33 = inttoptr i32 %a2 to i8*
  %v34 = icmp eq i8* %v30, %v32
  br i1 %v34, label %b15, label %b7

b7:                                               ; preds = %b6
  br label %b8

b8:                                               ; preds = %b12, %b7
  %v35 = phi i8* [ %v47, %b12 ], [ %v30, %b7 ]
  %v36 = phi i8* [ %v48, %b12 ], [ %v6, %b7 ]
  %v37 = phi i8* [ %v46, %b12 ], [ %v7, %b7 ]
  %v38 = icmp eq i8* %v37, %v33
  br i1 %v38, label %b13, label %b9

b9:                                               ; preds = %b8
  %v39 = load i8, i8* %v37, align 1, !tbaa !4
  %v40 = load i8, i8* %v35, align 1, !tbaa !4
  %v41 = call zeroext i1 %a6(i8 zeroext %v39, i8 zeroext %v40)
  br i1 %v41, label %b10, label %b11

b10:                                              ; preds = %b9
  %v42 = load i8, i8* %v37, align 1, !tbaa !4
  store i8 %v42, i8* %v36, align 1, !tbaa !4
  %v43 = getelementptr inbounds i8, i8* %v37, i32 1
  br label %b12

b11:                                              ; preds = %b9
  %v44 = load i8, i8* %v35, align 1, !tbaa !4
  store i8 %v44, i8* %v36, align 1, !tbaa !4
  %v45 = getelementptr inbounds i8, i8* %v35, i32 1
  br label %b12

b12:                                              ; preds = %b11, %b10
  %v46 = phi i8* [ %v43, %b10 ], [ %v37, %b11 ]
  %v47 = phi i8* [ %v35, %b10 ], [ %v45, %b11 ]
  %v48 = getelementptr inbounds i8, i8* %v36, i32 1
  %v49 = icmp eq i8* %v47, %v32
  br i1 %v49, label %b14, label %b8

b13:                                              ; preds = %b8
  call void @f9(i8* %v35, i8* %v36, i8* %v32)
  br label %b43

b14:                                              ; preds = %b12
  br label %b15

b15:                                              ; preds = %b14, %b6
  %v50 = phi i8* [ %v7, %b6 ], [ %v46, %b14 ]
  %v51 = phi i8* [ %v6, %b6 ], [ %v48, %b14 ]
  %v52 = icmp eq i8* %v50, %v33
  br i1 %v52, label %b43, label %b16

b16:                                              ; preds = %b15
  br label %b17

b17:                                              ; preds = %b17, %b16
  %v53 = phi i8* [ %v56, %b17 ], [ %v51, %b16 ]
  %v54 = phi i8* [ %v57, %b17 ], [ %v50, %b16 ]
  %v55 = load i8, i8* %v54, align 1, !tbaa !4
  store i8 %v55, i8* %v53, align 1, !tbaa !4
  %v56 = getelementptr inbounds i8, i8* %v53, i32 1
  %v57 = getelementptr inbounds i8, i8* %v54, i32 1
  %v58 = icmp eq i8* %v57, %v33
  br i1 %v58, label %b42, label %b17

b18:                                              ; preds = %b3, %b2
  %v59 = call i32 @f0(%s.0* %a5)
  %v60 = icmp slt i32 %v59, %a4
  br i1 %v60, label %b33, label %b19

b19:                                              ; preds = %b18
  %v61 = getelementptr inbounds %s.0, %s.0* %a5, i32 0, i32 1
  %v62 = load %s.1*, %s.1** %v61, align 4, !tbaa !0
  %v63 = getelementptr inbounds %s.1, %s.1* %v62, i32 0, i32 0
  %v64 = load i8*, i8** %v63, align 4, !tbaa !0
  %v65 = getelementptr inbounds %s.1, %s.1* %v62, i32 0, i32 1
  store i8* %v64, i8** %v65, align 4, !tbaa !0
  %v66 = bitcast %s.0* %v5 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %v66, i8 0, i64 16, i1 false)
  %v67 = load %s.1*, %s.1** %v61, align 4, !tbaa !0
  %v68 = getelementptr inbounds %s.0, %s.0* %v5, i32 0, i32 1
  store %s.1* %v67, %s.1** %v68, align 4, !tbaa !0
  %v69 = bitcast %s.0* %v0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %v69)
  call void @llvm.memset.p0i8.i64(i8* align 4 %v69, i8 0, i64 16, i1 false)
  %v70 = getelementptr inbounds %s.0, %s.0* %v0, i32 0, i32 1
  store %s.1* %v67, %s.1** %v70, align 4, !tbaa !0
  %v71 = inttoptr i32 %a2 to i8*
  %v72 = icmp eq i8* %v7, %v71
  br i1 %v72, label %b21, label %b20

b20:                                              ; preds = %b19
  call void @f8(i8* %v7, %s.0* %v0, i8* %v71)
  %v73 = load %s.1*, %s.1** %v70, align 4, !tbaa !0
  br label %b21

b21:                                              ; preds = %b20, %b19
  %v74 = phi %s.1* [ %v73, %b20 ], [ %v67, %b19 ]
  %v75 = bitcast %s.0* %v4 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %v75, i8 0, i64 16, i1 false)
  %v76 = getelementptr inbounds %s.0, %s.0* %v4, i32 0, i32 1
  store %s.1* %v74, %s.1** %v76, align 4, !tbaa !0
  call void @f1(%s.0* %v0) #0
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %v69)
  call void @f1(%s.0* %v4) #0
  call void @f1(%s.0* %v5) #0
  %v77 = load %s.1*, %s.1** %v61, align 4, !tbaa !0
  %v78 = getelementptr inbounds %s.1, %s.1* %v77, i32 0, i32 0
  %v79 = load i8*, i8** %v78, align 4, !tbaa !0
  %v80 = getelementptr inbounds %s.1, %s.1* %v77, i32 0, i32 1
  %v81 = load i8*, i8** %v80, align 4, !tbaa !0
  %v82 = icmp eq i8* %v6, %v7
  br i1 %v82, label %b25, label %b22

b22:                                              ; preds = %b21
  br label %b23

b23:                                              ; preds = %b31, %b22
  %v83 = phi i8* [ %v100, %b31 ], [ %v81, %b22 ]
  %v84 = phi i8* [ %v111, %b31 ], [ %v71, %b22 ]
  %v85 = phi i8* [ %v86, %b31 ], [ %v7, %b22 ]
  %v86 = getelementptr inbounds i8, i8* %v85, i32 -1
  %v87 = icmp eq i8* %v83, %v79
  br i1 %v87, label %b28, label %b24

b24:                                              ; preds = %b23
  br label %b30

b25:                                              ; preds = %b31, %b21
  %v88 = phi i8* [ %v81, %b21 ], [ %v100, %b31 ]
  %v89 = phi i8* [ %v71, %b21 ], [ %v111, %b31 ]
  %v90 = icmp eq i8* %v88, %v79
  br i1 %v90, label %b43, label %b26

b26:                                              ; preds = %b25
  br label %b27

b27:                                              ; preds = %b27, %b26
  %v91 = phi i8* [ %v93, %b27 ], [ %v88, %b26 ]
  %v92 = phi i8* [ %v95, %b27 ], [ %v89, %b26 ]
  %v93 = getelementptr inbounds i8, i8* %v91, i32 -1
  %v94 = load i8, i8* %v93, align 1, !tbaa !4
  %v95 = getelementptr inbounds i8, i8* %v92, i32 -1
  store i8 %v94, i8* %v95, align 1, !tbaa !4
  %v96 = icmp eq i8* %v93, %v79
  br i1 %v96, label %b41, label %b27

b28:                                              ; preds = %b31, %b23
  %v97 = phi i8* [ %v111, %b31 ], [ %v84, %b23 ]
  %v98 = icmp eq i8* %v6, %v85
  br i1 %v98, label %b43, label %b29

b29:                                              ; preds = %b28
  call void @f6(i8* %v97, i8* %v85, i8* %v6)
  br label %b43

b30:                                              ; preds = %b31, %b24
  %v99 = phi i8* [ %v111, %b31 ], [ %v84, %b24 ]
  %v100 = phi i8* [ %v101, %b31 ], [ %v83, %b24 ]
  %v101 = getelementptr inbounds i8, i8* %v100, i32 -1
  %v102 = load i8, i8* %v101, align 1, !tbaa !4
  %v103 = load i8, i8* %v86, align 1, !tbaa !4
  %v104 = call zeroext i1 %a6(i8 zeroext %v102, i8 zeroext %v103)
  br i1 %v104, label %b31, label %b32

b31:                                              ; preds = %b32, %b30
  %v105 = phi i8* [ %v101, %b32 ], [ %v86, %b30 ]
  %v106 = phi i8* [ %v101, %b32 ], [ %v6, %b30 ]
  %v107 = phi i8* [ %v79, %b32 ], [ %v86, %b30 ]
  %v108 = phi i8* [ blockaddress(@f2, %b30), %b32 ], [ blockaddress(@f2, %b23), %b30 ]
  %v109 = phi i8* [ blockaddress(@f2, %b28), %b32 ], [ blockaddress(@f2, %b25), %b30 ]
  %v110 = load i8, i8* %v105, align 1, !tbaa !4
  %v111 = getelementptr inbounds i8, i8* %v99, i32 -1
  store i8 %v110, i8* %v111, align 1, !tbaa !4
  %v112 = icmp eq i8* %v106, %v107
  %v113 = select i1 %v112, i8* %v109, i8* %v108
  indirectbr i8* %v113, [label %b25, label %b28, label %b23, label %b30]

b32:                                              ; preds = %b30
  br label %b31

b33:                                              ; preds = %b18
  br i1 %v10, label %b34, label %b37

b34:                                              ; preds = %b33
  %v114 = sdiv i32 %a3, 2
  %v115 = getelementptr inbounds i8, i8* %v6, i32 %v114
  %v116 = sub i32 %a2, %a1
  %v117 = icmp sgt i32 %v116, 0
  br i1 %v117, label %b35, label %b36

b35:                                              ; preds = %b34
  %v118 = call i8* @f5(i8* %v7, i32 %v116, i8* %v115, i1 (i8, i8)* %a6)
  br label %b36

b36:                                              ; preds = %b35, %b34
  %v119 = phi i8* [ %v7, %b34 ], [ %v118, %b35 ]
  %v120 = ptrtoint i8* %v119 to i32
  %v121 = sub i32 %v120, %a1
  br label %b40

b37:                                              ; preds = %b33
  %v122 = sdiv i32 %a4, 2
  %v123 = getelementptr inbounds i8, i8* %v7, i32 %v122
  %v124 = sub i32 %a1, %a0
  %v125 = icmp sgt i32 %v124, 0
  br i1 %v125, label %b38, label %b39

b38:                                              ; preds = %b37
  %v126 = call i8* @f4(i8* %v6, i32 %v124, i8* %v123, i1 (i8, i8)* %a6)
  br label %b39

b39:                                              ; preds = %b38, %b37
  %v127 = phi i8* [ %v6, %b37 ], [ %v126, %b38 ]
  %v128 = ptrtoint i8* %v127 to i32
  %v129 = sub i32 %v128, %a0
  br label %b40

b40:                                              ; preds = %b39, %b36
  %v130 = phi i8* [ %v127, %b39 ], [ %v115, %b36 ]
  %v131 = phi i8* [ %v123, %b39 ], [ %v119, %b36 ]
  %v132 = phi i32 [ %v129, %b39 ], [ %v114, %b36 ]
  %v133 = phi i32 [ %v122, %b39 ], [ %v121, %b36 ]
  %v134 = sub nsw i32 %a3, %v132
  %v135 = ptrtoint i8* %v130 to i32
  %v136 = ptrtoint i8* %v131 to i32
  %v137 = call i32 @f3(i32 %v135, i32 %a1, i32 %v136, i32 %v134, i32 %v133, %s.0* %a5)
  call void @f2(i32 %a0, i32 %v135, i32 %v137, i32 %v132, i32 %v133, %s.0* %a5, i1 (i8, i8)* %a6)
  %v138 = sub nsw i32 %a4, %v133
  call void @f2(i32 %v137, i32 %v136, i32 %a2, i32 %v134, i32 %v138, %s.0* %a5, i1 (i8, i8)* %a6)
  br label %b43

b41:                                              ; preds = %b27
  br label %b43

b42:                                              ; preds = %b17
  br label %b43

b43:                                              ; preds = %b42, %b41, %b40, %b29, %b28, %b25, %b15, %b13, %b1
  ret void
}

; Function Attrs: inlinehint
declare i32 @f3(i32, i32, i32, i32, i32, %s.0* nocapture) #1

; Function Attrs: inlinehint
declare i8* @f4(i8*, i32, i8*, i1 (i8, i8)*) #1

; Function Attrs: inlinehint
declare i8* @f5(i8*, i32, i8*, i1 (i8, i8)*) #1

; Function Attrs: inlinehint
declare void @f6(i8*, i8*, i8*) #1

; Function Attrs: inlinehint
declare void @f7(i8*, i8*, i1 (i8, i8)*) #1

; Function Attrs: inlinehint
declare void @f8(i8*, %s.0*, i8*) #1

; Function Attrs: inlinehint
declare void @f9(i8*, i8*, i8*) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #2

attributes #0 = { nounwind }
attributes #1 = { inlinehint }
attributes #2 = { argmemonly nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
