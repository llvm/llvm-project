; RUN: llc -march=hexagon -machine-sink-split=0 < %s
; REQUIRES: asserts

target triple = "hexagon-unknown-linux-gnu"

%s.0 = type { double, double, double, double, double, double, i32, double, double, double, double, i8*, i8, [9 x i8], double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, [200 x i8*], [32 x i8*], [32 x i8], i32 }

@g0 = external unnamed_addr constant [6 x i8], align 8

; Function Attrs: nounwind
define i32 @f0(double %a0) #0 {
b0:
  %v0 = call double bitcast (double (...)* @f1 to double (i8*)*)(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @g0, i32 0, i32 0)) #0
  %v1 = call i32 bitcast (i32 (...)* @f2 to i32 ()*)() #0
  %v2 = call i8* @f3(i32 undef)
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b2
  unreachable

b4:                                               ; preds = %b2
  %v3 = mul i32 %v1, 12
  br i1 undef, label %b5, label %b6

b5:                                               ; preds = %b4
  ret i32 0

b6:                                               ; preds = %b4
  %v4 = call i32 bitcast (i32 (...)* @f2 to i32 ()*)() #0
  br i1 undef, label %b7, label %b24

b7:                                               ; preds = %b6
  switch i32 undef, label %b8 [
    i32 0, label %b15
    i32 1, label %b14
    i32 2, label %b13
    i32 3, label %b12
    i32 4, label %b11
    i32 5, label %b10
    i32 6, label %b9
  ]

b8:                                               ; preds = %b7
  unreachable

b9:                                               ; preds = %b7
  br label %b10

b10:                                              ; preds = %b9, %b7
  unreachable

b11:                                              ; preds = %b7
  unreachable

b12:                                              ; preds = %b7
  br label %b13

b13:                                              ; preds = %b12, %b7
  unreachable

b14:                                              ; preds = %b7
  %v5 = call %s.0* bitcast (%s.0* (...)* @f4 to %s.0* (i32)*)(i32 0) #0
  %v6 = icmp ult i32 %v4, 8
  br i1 %v6, label %b16, label %b15

b15:                                              ; preds = %b14, %b7
  unreachable

b16:                                              ; preds = %b14
  %v7 = and i32 %v4, 3
  br i1 undef, label %b17, label %b18

b17:                                              ; preds = %b16
  br i1 undef, label %b19, label %b18

b18:                                              ; preds = %b18, %b17, %b16
  %v8 = phi i32 [ %v10, %b18 ], [ 0, %b16 ], [ undef, %b17 ]
  %v9 = shl i32 %v8, 5
  %v10 = add nsw i32 %v8, 4
  %v11 = icmp eq i32 %v10, %v4
  br i1 %v11, label %b19, label %b18

b19:                                              ; preds = %b18, %b17
  br i1 undef, label %b20, label %b23

b20:                                              ; preds = %b19
  %v12 = icmp eq i32 %v7, 2
  br i1 %v12, label %b21, label %b22

b21:                                              ; preds = %b20
  %v13 = getelementptr i8, i8* %v2, i32 0
  %v14 = bitcast i8* %v13 to double*
  %v15 = or i32 0, 16
  %v16 = getelementptr i8, i8* %v2, i32 %v15
  %v17 = bitcast i8* %v16 to double*
  %v18 = load double, double* undef, align 8, !tbaa !0
  %v19 = fcmp olt double -1.000000e+11, %v18
  %v20 = select i1 %v19, double %v18, double -1.000000e+11
  %v21 = load double, double* %v14, align 8, !tbaa !0
  %v22 = fcmp olt double -1.000000e+11, %v21
  %v23 = select i1 %v22, double %v21, double -1.000000e+11
  %v24 = load double, double* %v17, align 8, !tbaa !0
  %v25 = fcmp olt double -1.000000e+11, %v24
  %v26 = select i1 %v25, double %v24, double -1.000000e+11
  %v27 = fcmp ogt double 1.000000e+11, %v18
  %v28 = select i1 %v27, double %v18, double 1.000000e+11
  %v29 = fcmp ogt double 1.000000e+11, %v21
  %v30 = select i1 %v29, double %v21, double 1.000000e+11
  %v31 = fcmp ogt double 1.000000e+11, %v24
  %v32 = select i1 %v31, double %v24, double 1.000000e+11
  %v33 = add i32 0, 1
  %v34 = getelementptr i8, i8* %v2, i32 32
  br label %b22

b22:                                              ; preds = %b21, %b20
  %v35 = phi double [ %v20, %b21 ], [ -1.000000e+11, %b20 ]
  %v36 = phi double [ %v28, %b21 ], [ 1.000000e+11, %b20 ]
  %v37 = phi double [ %v23, %b21 ], [ -1.000000e+11, %b20 ]
  %v38 = phi double [ %v30, %b21 ], [ 1.000000e+11, %b20 ]
  %v39 = phi double [ %v26, %b21 ], [ -1.000000e+11, %b20 ]
  %v40 = phi double [ %v32, %b21 ], [ 1.000000e+11, %b20 ]
  %v41 = phi i8* [ %v34, %b21 ], [ %v2, %b20 ]
  %v42 = phi i32 [ %v33, %b21 ], [ 0, %b20 ]
  %v43 = shl nsw i32 %v42, 5
  %v44 = bitcast i8* %v41 to double*
  %v45 = or i32 %v43, 8
  %v46 = getelementptr i8, i8* %v2, i32 %v45
  %v47 = bitcast i8* %v46 to double*
  %v48 = load double, double* %v44, align 8, !tbaa !0
  %v49 = select i1 undef, double %v48, double %v35
  %v50 = load double, double* %v47, align 8, !tbaa !0
  %v51 = fcmp olt double %v37, %v50
  %v52 = select i1 %v51, double %v50, double %v37
  %v53 = load double, double* undef, align 8, !tbaa !0
  %v54 = fcmp olt double %v39, %v53
  %v55 = select i1 %v54, double %v53, double %v39
  %v56 = fcmp ogt double %v36, %v48
  %v57 = select i1 %v56, double %v48, double %v36
  %v58 = fcmp ogt double %v38, %v50
  %v59 = select i1 %v58, double %v50, double %v38
  %v60 = select i1 undef, double %v53, double %v40
  %v61 = add i32 %v42, 1
  br i1 undef, label %b24, label %b23

b23:                                              ; preds = %b23, %b22, %b19
  %v62 = phi double [ %v79, %b23 ], [ 1.000000e+11, %b19 ], [ %v57, %b22 ]
  %v63 = phi double [ %v81, %b23 ], [ 1.000000e+11, %b19 ], [ %v59, %b22 ]
  %v64 = phi i32 [ %v82, %b23 ], [ 0, %b19 ], [ %v61, %b22 ]
  %v65 = shl i32 %v64, 5
  %v66 = load double, double* undef, align 8, !tbaa !0
  %v67 = load double, double* undef, align 8, !tbaa !0
  %v68 = select i1 undef, double %v66, double %v62
  %v69 = select i1 undef, double %v67, double %v63
  %v70 = load double, double* undef, align 8, !tbaa !0
  %v71 = select i1 false, double 0.000000e+00, double %v68
  %v72 = select i1 undef, double %v70, double %v69
  %v73 = bitcast i8* undef to double*
  %v74 = load double, double* undef, align 8, !tbaa !0
  %v75 = fcmp ogt double %v71, 0.000000e+00
  %v76 = select i1 %v75, double 0.000000e+00, double %v71
  %v77 = select i1 undef, double %v74, double %v72
  %v78 = load double, double* undef, align 8, !tbaa !0
  %v79 = select i1 undef, double %v78, double %v76
  %v80 = fcmp ogt double %v77, 0.000000e+00
  %v81 = select i1 %v80, double 0.000000e+00, double %v77
  %v82 = add i32 %v64, 4
  %v83 = icmp eq i32 %v82, %v4
  br i1 %v83, label %b24, label %b23

b24:                                              ; preds = %b23, %b22, %b6
  %v84 = phi double [ -1.000000e+11, %b6 ], [ %v49, %b22 ], [ undef, %b23 ]
  %v85 = phi double [ -1.000000e+11, %b6 ], [ %v52, %b22 ], [ 0.000000e+00, %b23 ]
  %v86 = phi double [ -1.000000e+11, %b6 ], [ %v55, %b22 ], [ 0.000000e+00, %b23 ]
  %v87 = phi double [ 1.000000e+11, %b6 ], [ %v60, %b22 ], [ undef, %b23 ]
  %v88 = fsub double %v84, undef
  %v89 = fsub double %v85, undef
  %v90 = fadd double undef, 1.000000e+00
  %v91 = fptosi double %v90 to i32
  %v92 = fsub double %v86, %v87
  %v93 = fdiv double %v92, %v0
  %v94 = fadd double %v93, 1.000000e+00
  %v95 = fptosi double %v94 to i32
  br i1 undef, label %b25, label %b27

b25:                                              ; preds = %b24
  %v96 = fdiv double %v88, 0.000000e+00
  %v97 = fadd double %v96, 1.000000e+00
  %v98 = fptosi double %v97 to i32
  %v99 = fdiv double %v89, 0.000000e+00
  %v100 = fadd double %v99, 1.000000e+00
  %v101 = fptosi double %v100 to i32
  %v102 = fadd double undef, 1.000000e+00
  %v103 = fptosi double %v102 to i32
  %v104 = call i8* @f3(i32 undef)
  br i1 false, label %b26, label %b27

b26:                                              ; preds = %b25
  unreachable

b27:                                              ; preds = %b25, %b24
  %v105 = phi i8* [ %v104, %b25 ], [ undef, %b24 ]
  %v106 = phi i32 [ %v103, %b25 ], [ %v95, %b24 ]
  %v107 = phi i32 [ %v101, %b25 ], [ %v91, %b24 ]
  %v108 = phi i32 [ %v98, %b25 ], [ undef, %b24 ]
  %v109 = phi double [ 0.000000e+00, %b25 ], [ %v0, %b24 ]
  %v110 = mul i32 %v108, 232
  %v111 = icmp sgt i32 %v106, 0
  %v112 = mul i32 %v107, 232
  %v113 = mul i32 %v112, %v108
  %v114 = fmul double %v109, 5.000000e-01
  %v115 = and i32 %v106, 3
  %v116 = icmp ult i32 %v106, 4
  br label %b28

b28:                                              ; preds = %b35, %b27
  %v117 = phi i32 [ %v146, %b35 ], [ 0, %b27 ]
  %v118 = mul i32 %v117, 232
  br i1 undef, label %b29, label %b35

b29:                                              ; preds = %b28
  %v119 = add i32 %v118, 8
  %v120 = add i32 %v118, 16
  br i1 %v111, label %b30, label %b35

b30:                                              ; preds = %b34, %b29
  %v121 = phi i32 [ %v144, %b34 ], [ 0, %b29 ]
  %v122 = mul i32 %v110, %v121
  %v123 = add i32 %v119, %v122
  %v124 = add i32 %v120, %v122
  %v125 = sitofp i32 %v121 to double
  %v126 = fmul double %v125, %v109
  %v127 = fadd double %v126, %v114
  %v128 = fadd double %v127, undef
  switch i32 %v115, label %b33 [
    i32 2, label %b31
    i32 1, label %b32
  ]

b31:                                              ; preds = %b30
  %v129 = add i32 %v123, 0
  %v130 = getelementptr i8, i8* %v105, i32 %v129
  %v131 = bitcast i8* %v130 to double*
  store double %v128, double* %v131, align 8, !tbaa !0
  br label %b32

b32:                                              ; preds = %b31, %b30
  %v132 = add nsw i32 0, 1
  br i1 %v116, label %b34, label %b33

b33:                                              ; preds = %b33, %b32, %b30
  %v133 = phi i32 [ %v142, %b33 ], [ 0, %b30 ], [ %v132, %b32 ]
  %v134 = mul i32 %v113, %v133
  %v135 = add i32 %v124, %v134
  %v136 = getelementptr i8, i8* %v105, i32 %v135
  %v137 = bitcast i8* %v136 to double*
  %v138 = sitofp i32 %v133 to double
  store double undef, double* %v137, align 8, !tbaa !0
  %v139 = fmul double undef, %v109
  %v140 = fadd double %v139, %v114
  %v141 = fadd double %v140, %v87
  store double %v141, double* undef, align 8, !tbaa !0
  %v142 = add nsw i32 %v133, 4
  %v143 = icmp eq i32 %v142, %v106
  br i1 %v143, label %b34, label %b33

b34:                                              ; preds = %b33, %b32
  %v144 = add i32 %v121, 1
  %v145 = icmp eq i32 %v144, %v107
  br i1 %v145, label %b35, label %b30

b35:                                              ; preds = %b34, %b29, %b28
  %v146 = add i32 %v117, 1
  br label %b28
}

declare double @f1(...)

declare i32 @f2(...)

; Function Attrs: nounwind
declare noalias i8* @f3(i32) #0

declare %s.0* @f4(...)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
