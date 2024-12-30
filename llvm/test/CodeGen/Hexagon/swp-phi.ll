; RUN: llc -mtriple=hexagon -enable-unsafe-fp-math -enable-pipeliner \
; RUN:     -pipeliner-prune-deps=false -stats -o /dev/null < %s
; REQUIRES: asserts

; Test that checks we dont crash when SWP a loop with lots of Phis, and
; the Phi operand refer to Phis from the same loop.

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0, ptr nocapture %a1) #0 {
b0:
  %v0 = alloca [400 x float], align 4
  %v1 = getelementptr inbounds float, ptr %a1, i32 1
  %v2 = getelementptr inbounds float, ptr %a1, i32 2
  %v3 = getelementptr inbounds float, ptr %a1, i32 3
  %v4 = getelementptr inbounds float, ptr %a1, i32 4
  %v5 = getelementptr inbounds float, ptr %a1, i32 5
  %v6 = getelementptr inbounds float, ptr %a1, i32 6
  %v7 = getelementptr inbounds float, ptr %a1, i32 7
  %v8 = getelementptr inbounds float, ptr %a1, i32 8
  %v9 = getelementptr inbounds float, ptr %a1, i32 9
  %v10 = getelementptr inbounds float, ptr %a1, i32 10
  %v11 = getelementptr inbounds float, ptr %a1, i32 11
  %v12 = getelementptr inbounds float, ptr %a1, i32 12
  %v13 = getelementptr inbounds float, ptr %a1, i32 13
  %v14 = getelementptr inbounds float, ptr %a1, i32 14
  %v15 = getelementptr inbounds float, ptr %a1, i32 15
  %v16 = getelementptr inbounds float, ptr %a1, i32 16
  %v17 = load float, ptr %a1, align 4
  %v18 = load float, ptr %v1, align 4
  %v19 = load float, ptr %v2, align 4
  %v20 = load float, ptr %v3, align 4
  %v21 = load float, ptr %v4, align 4
  %v22 = load float, ptr %v5, align 4
  %v23 = load float, ptr %v6, align 4
  %v24 = load float, ptr %v7, align 4
  %v25 = load float, ptr %v8, align 4
  %v26 = load float, ptr %v9, align 4
  %v27 = load float, ptr %v10, align 4
  %v28 = load float, ptr %v11, align 4
  %v29 = load float, ptr %v12, align 4
  %v30 = load float, ptr %v13, align 4
  %v31 = load float, ptr %v14, align 4
  %v32 = load float, ptr %v15, align 4
  %v33 = load float, ptr %v16, align 4
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v34 = phi float [ undef, %b0 ], [ %v103, %b1 ]
  %v35 = phi float [ undef, %b0 ], [ %v34, %b1 ]
  %v36 = phi float [ undef, %b0 ], [ %v35, %b1 ]
  %v37 = phi float [ undef, %b0 ], [ %v36, %b1 ]
  %v38 = phi float [ undef, %b0 ], [ %v37, %b1 ]
  %v39 = phi float [ undef, %b0 ], [ %v38, %b1 ]
  %v40 = phi float [ undef, %b0 ], [ %v39, %b1 ]
  %v41 = phi float [ undef, %b0 ], [ %v40, %b1 ]
  %v42 = phi float [ undef, %b0 ], [ %v41, %b1 ]
  %v43 = phi float [ undef, %b0 ], [ %v42, %b1 ]
  %v44 = phi float [ undef, %b0 ], [ %v43, %b1 ]
  %v45 = phi float [ undef, %b0 ], [ %v44, %b1 ]
  %v46 = phi float [ undef, %b0 ], [ %v45, %b1 ]
  %v47 = phi float [ undef, %b0 ], [ %v46, %b1 ]
  %v48 = phi float [ undef, %b0 ], [ %v47, %b1 ]
  %v49 = phi float [ undef, %b0 ], [ %v48, %b1 ]
  %v50 = phi float [ %v33, %b0 ], [ %v105, %b1 ]
  %v51 = phi float [ %v32, %b0 ], [ %v100, %b1 ]
  %v52 = phi float [ %v31, %b0 ], [ %v98, %b1 ]
  %v53 = phi float [ %v30, %b0 ], [ %v96, %b1 ]
  %v54 = phi float [ %v29, %b0 ], [ %v94, %b1 ]
  %v55 = phi float [ %v28, %b0 ], [ %v92, %b1 ]
  %v56 = phi float [ %v27, %b0 ], [ %v90, %b1 ]
  %v57 = phi float [ %v26, %b0 ], [ %v88, %b1 ]
  %v58 = phi float [ %v25, %b0 ], [ %v86, %b1 ]
  %v59 = phi float [ %v24, %b0 ], [ %v84, %b1 ]
  %v60 = phi float [ %v23, %b0 ], [ %v82, %b1 ]
  %v61 = phi float [ %v22, %b0 ], [ %v80, %b1 ]
  %v62 = phi float [ %v21, %b0 ], [ %v78, %b1 ]
  %v63 = phi float [ %v20, %b0 ], [ %v76, %b1 ]
  %v64 = phi float [ %v19, %b0 ], [ %v74, %b1 ]
  %v65 = phi float [ %v18, %b0 ], [ %v72, %b1 ]
  %v66 = phi float [ %v17, %b0 ], [ %v69, %b1 ]
  %v67 = phi i32 [ 0, %b0 ], [ %v70, %b1 ]
  %v68 = fmul float %v49, %v49
  %v69 = fadd float %v66, %v68
  %v70 = add nsw i32 %v67, 1
  %v71 = fmul float %v49, %v48
  %v72 = fadd float %v65, %v71
  %v73 = fmul float %v49, %v47
  %v74 = fadd float %v64, %v73
  %v75 = fmul float %v49, %v46
  %v76 = fadd float %v63, %v75
  %v77 = fmul float %v49, %v45
  %v78 = fadd float %v62, %v77
  %v79 = fmul float %v49, %v44
  %v80 = fadd float %v61, %v79
  %v81 = fmul float %v49, %v43
  %v82 = fadd float %v60, %v81
  %v83 = fmul float %v49, %v42
  %v84 = fadd float %v59, %v83
  %v85 = fmul float %v49, %v41
  %v86 = fadd float %v58, %v85
  %v87 = fmul float %v49, %v40
  %v88 = fadd float %v57, %v87
  %v89 = fmul float %v49, %v39
  %v90 = fadd float %v56, %v89
  %v91 = fmul float %v49, %v38
  %v92 = fadd float %v55, %v91
  %v93 = fmul float %v49, %v37
  %v94 = fadd float %v54, %v93
  %v95 = fmul float %v49, %v36
  %v96 = fadd float %v53, %v95
  %v97 = fmul float %v49, %v35
  %v98 = fadd float %v52, %v97
  %v99 = fmul float %v49, %v34
  %v100 = fadd float %v51, %v99
  %v101 = add nsw i32 %v67, 16
  %v102 = getelementptr inbounds [400 x float], ptr %v0, i32 0, i32 %v101
  %v103 = load float, ptr %v102, align 4, !tbaa !0
  %v104 = fmul float %v49, %v103
  %v105 = fadd float %v50, %v104
  %v106 = icmp eq i32 %v70, 384
  br i1 %v106, label %b2, label %b1

b2:                                               ; preds = %b1
  store float %v69, ptr %a1, align 4
  store float %v72, ptr %v1, align 4
  store float %v74, ptr %v2, align 4
  store float %v76, ptr %v3, align 4
  store float %v78, ptr %v4, align 4
  store float %v80, ptr %v5, align 4
  store float %v82, ptr %v6, align 4
  store float %v84, ptr %v7, align 4
  store float %v86, ptr %v8, align 4
  store float %v88, ptr %v9, align 4
  store float %v90, ptr %v10, align 4
  store float %v92, ptr %v11, align 4
  store float %v94, ptr %v12, align 4
  store float %v96, ptr %v13, align 4
  store float %v98, ptr %v14, align 4
  store float %v100, ptr %v15, align 4
  store float %v105, ptr %v16, align 4
  %v107 = fcmp olt float %v69, 1.000000e+00
  br i1 %v107, label %b3, label %b4

b3:                                               ; preds = %b2
  store float 1.000000e+00, ptr %a1, align 4, !tbaa !0
  br label %b4

b4:                                               ; preds = %b3, %b2
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
