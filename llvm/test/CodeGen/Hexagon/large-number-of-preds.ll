; RUN: llc -O3 -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon-unknown--elf"

@g0 = external global ptr

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0, ptr nocapture %a1, ptr %a2) #0 {
b0:
  %v0 = alloca [64 x float], align 16
  %v1 = alloca [8 x float], align 8
  call void @llvm.lifetime.start.p0(i64 256, ptr %v0) #2
  %v3 = load float, ptr %a0, align 4, !tbaa !0
  %v4 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 35
  store float %v3, ptr %v4, align 4, !tbaa !0
  store float %v3, ptr %v0, align 16, !tbaa !0
  %v6 = getelementptr inbounds float, ptr %a0, i32 1
  %v7 = load float, ptr %v6, align 4, !tbaa !0
  %v8 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 36
  store float %v7, ptr %v8, align 16, !tbaa !0
  %v9 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 1
  store float %v7, ptr %v9, align 4, !tbaa !0
  %v10 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 37
  store float 1.000000e+00, ptr %v10, align 4, !tbaa !0
  %v11 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 2
  store float 1.000000e+00, ptr %v11, align 8, !tbaa !0
  %v12 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 34
  store float 0.000000e+00, ptr %v12, align 8, !tbaa !0
  %v13 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 33
  store float 0.000000e+00, ptr %v13, align 4, !tbaa !0
  %v14 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 32
  store float 0.000000e+00, ptr %v14, align 16, !tbaa !0
  %v15 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 5
  store float 0.000000e+00, ptr %v15, align 4, !tbaa !0
  %v16 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 4
  store float 0.000000e+00, ptr %v16, align 16, !tbaa !0
  %v17 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 3
  store float 0.000000e+00, ptr %v17, align 4, !tbaa !0
  %v18 = load float, ptr %a1, align 4, !tbaa !0
  %v19 = fmul float %v3, %v18
  %v20 = fsub float -0.000000e+00, %v19
  %v21 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 6
  store float %v20, ptr %v21, align 8, !tbaa !0
  %v22 = fmul float %v7, %v18
  %v23 = fsub float -0.000000e+00, %v22
  %v24 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 7
  store float %v23, ptr %v24, align 4, !tbaa !0
  %v25 = getelementptr inbounds float, ptr %a1, i32 1
  %v26 = load float, ptr %v25, align 4, !tbaa !0
  %v27 = fmul float %v3, %v26
  %v28 = fsub float -0.000000e+00, %v27
  %v29 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 38
  store float %v28, ptr %v29, align 8, !tbaa !0
  %v30 = fmul float %v7, %v26
  %v31 = fsub float -0.000000e+00, %v30
  %v32 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 39
  store float %v31, ptr %v32, align 4, !tbaa !0
  store float %v18, ptr %v1, align 8, !tbaa !0
  %v34 = getelementptr inbounds [8 x float], ptr %v1, i32 0, i32 4
  store float %v26, ptr %v34, align 8, !tbaa !0
  %v35 = getelementptr float, ptr %a0, i32 2
  %v36 = getelementptr float, ptr %a1, i32 2
  %v37 = load float, ptr %v35, align 4, !tbaa !0
  %v38 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 43
  store float %v37, ptr %v38, align 4, !tbaa !0
  %v39 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 8
  store float %v37, ptr %v39, align 16, !tbaa !0
  %v40 = getelementptr inbounds float, ptr %a0, i32 3
  %v41 = load float, ptr %v40, align 4, !tbaa !0
  %v42 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 44
  store float %v41, ptr %v42, align 16, !tbaa !0
  %v43 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 9
  store float %v41, ptr %v43, align 4, !tbaa !0
  %v44 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 45
  store float 1.000000e+00, ptr %v44, align 4, !tbaa !0
  %v45 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 10
  store float 1.000000e+00, ptr %v45, align 8, !tbaa !0
  %v46 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 42
  store float 0.000000e+00, ptr %v46, align 8, !tbaa !0
  %v47 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 41
  store float 0.000000e+00, ptr %v47, align 4, !tbaa !0
  %v48 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 40
  store float 0.000000e+00, ptr %v48, align 16, !tbaa !0
  %v49 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 13
  store float 0.000000e+00, ptr %v49, align 4, !tbaa !0
  %v50 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 12
  store float 0.000000e+00, ptr %v50, align 16, !tbaa !0
  %v51 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 11
  store float 0.000000e+00, ptr %v51, align 4, !tbaa !0
  %v52 = load float, ptr %v36, align 4, !tbaa !0
  %v53 = fmul float %v37, %v52
  %v54 = fsub float -0.000000e+00, %v53
  %v55 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 14
  store float %v54, ptr %v55, align 8, !tbaa !0
  %v56 = fmul float %v41, %v52
  %v57 = fsub float -0.000000e+00, %v56
  %v58 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 15
  store float %v57, ptr %v58, align 4, !tbaa !0
  %v59 = getelementptr inbounds float, ptr %a1, i32 3
  %v60 = load float, ptr %v59, align 4, !tbaa !0
  %v61 = fmul float %v37, %v60
  %v62 = fsub float -0.000000e+00, %v61
  %v63 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 46
  store float %v62, ptr %v63, align 8, !tbaa !0
  %v64 = fmul float %v41, %v60
  %v65 = fsub float -0.000000e+00, %v64
  %v66 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 47
  store float %v65, ptr %v66, align 4, !tbaa !0
  %v67 = getelementptr inbounds [8 x float], ptr %v1, i32 0, i32 1
  store float %v52, ptr %v67, align 4, !tbaa !0
  %v68 = getelementptr inbounds [8 x float], ptr %v1, i32 0, i32 5
  store float %v60, ptr %v68, align 4, !tbaa !0
  %v69 = getelementptr float, ptr %a0, i32 4
  %v70 = getelementptr float, ptr %a1, i32 4
  %v71 = load float, ptr %v69, align 4, !tbaa !0
  %v72 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 51
  store float %v71, ptr %v72, align 4, !tbaa !0
  %v73 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 16
  store float %v71, ptr %v73, align 16, !tbaa !0
  %v74 = getelementptr inbounds float, ptr %a0, i32 5
  %v75 = load float, ptr %v74, align 4, !tbaa !0
  %v76 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 52
  store float %v75, ptr %v76, align 16, !tbaa !0
  %v77 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 17
  store float %v75, ptr %v77, align 4, !tbaa !0
  %v78 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 53
  store float 1.000000e+00, ptr %v78, align 4, !tbaa !0
  %v79 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 18
  store float 1.000000e+00, ptr %v79, align 8, !tbaa !0
  %v80 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 50
  store float 0.000000e+00, ptr %v80, align 8, !tbaa !0
  %v81 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 49
  store float 0.000000e+00, ptr %v81, align 4, !tbaa !0
  %v82 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 48
  store float 0.000000e+00, ptr %v82, align 16, !tbaa !0
  %v83 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 21
  store float 0.000000e+00, ptr %v83, align 4, !tbaa !0
  %v84 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 20
  store float 0.000000e+00, ptr %v84, align 16, !tbaa !0
  %v85 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 19
  store float 0.000000e+00, ptr %v85, align 4, !tbaa !0
  %v86 = load float, ptr %v70, align 4, !tbaa !0
  %v87 = fmul float %v71, %v86
  %v88 = fsub float -0.000000e+00, %v87
  %v89 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 22
  store float %v88, ptr %v89, align 8, !tbaa !0
  %v90 = fmul float %v75, %v86
  %v91 = fsub float -0.000000e+00, %v90
  %v92 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 23
  store float %v91, ptr %v92, align 4, !tbaa !0
  %v93 = getelementptr inbounds float, ptr %a1, i32 5
  %v94 = load float, ptr %v93, align 4, !tbaa !0
  %v95 = fmul float %v71, %v94
  %v96 = fsub float -0.000000e+00, %v95
  %v97 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 54
  store float %v96, ptr %v97, align 8, !tbaa !0
  %v98 = fmul float %v75, %v94
  %v99 = fsub float -0.000000e+00, %v98
  %v100 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 55
  store float %v99, ptr %v100, align 4, !tbaa !0
  %v101 = getelementptr inbounds [8 x float], ptr %v1, i32 0, i32 2
  store float %v86, ptr %v101, align 8, !tbaa !0
  %v102 = getelementptr inbounds [8 x float], ptr %v1, i32 0, i32 6
  store float %v94, ptr %v102, align 8, !tbaa !0
  %v103 = getelementptr float, ptr %a0, i32 6
  %v104 = getelementptr float, ptr %a1, i32 6
  %v105 = load float, ptr %v103, align 4, !tbaa !0
  %v106 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 59
  store float %v105, ptr %v106, align 4, !tbaa !0
  %v107 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 24
  store float %v105, ptr %v107, align 16, !tbaa !0
  %v108 = getelementptr inbounds float, ptr %a0, i32 7
  %v109 = load float, ptr %v108, align 4, !tbaa !0
  %v110 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 60
  store float %v109, ptr %v110, align 16, !tbaa !0
  %v111 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 25
  store float %v109, ptr %v111, align 4, !tbaa !0
  %v112 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 61
  store float 1.000000e+00, ptr %v112, align 4, !tbaa !0
  %v113 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 26
  store float 1.000000e+00, ptr %v113, align 8, !tbaa !0
  %v114 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 58
  store float 0.000000e+00, ptr %v114, align 8, !tbaa !0
  %v115 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 57
  store float 0.000000e+00, ptr %v115, align 4, !tbaa !0
  %v116 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 56
  store float 0.000000e+00, ptr %v116, align 16, !tbaa !0
  %v117 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 29
  store float 0.000000e+00, ptr %v117, align 4, !tbaa !0
  %v118 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 28
  store float 0.000000e+00, ptr %v118, align 16, !tbaa !0
  %v119 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 27
  store float 0.000000e+00, ptr %v119, align 4, !tbaa !0
  %v120 = load float, ptr %v104, align 4, !tbaa !0
  %v121 = fmul float %v105, %v120
  %v122 = fsub float -0.000000e+00, %v121
  %v123 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 30
  store float %v122, ptr %v123, align 8, !tbaa !0
  %v124 = fmul float %v109, %v120
  %v125 = fsub float -0.000000e+00, %v124
  %v126 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 31
  store float %v125, ptr %v126, align 4, !tbaa !0
  %v127 = getelementptr inbounds float, ptr %a1, i32 7
  %v128 = load float, ptr %v127, align 4, !tbaa !0
  %v129 = fmul float %v105, %v128
  %v130 = fsub float -0.000000e+00, %v129
  %v131 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 62
  store float %v130, ptr %v131, align 8, !tbaa !0
  %v132 = fmul float %v109, %v128
  %v133 = fsub float -0.000000e+00, %v132
  %v134 = getelementptr inbounds [64 x float], ptr %v0, i32 0, i32 63
  store float %v133, ptr %v134, align 4, !tbaa !0
  %v135 = getelementptr inbounds [8 x float], ptr %v1, i32 0, i32 3
  store float %v120, ptr %v135, align 4, !tbaa !0
  %v136 = getelementptr inbounds [8 x float], ptr %v1, i32 0, i32 7
  store float %v128, ptr %v136, align 4, !tbaa !0
  %v137 = load ptr, ptr @g0, align 4, !tbaa !4
  %v138 = load ptr, ptr %v137, align 4, !tbaa !4
  call void %v138(ptr %v0, i32 8, i32 8, ptr %v1, ptr %a2) #2
  %v139 = getelementptr inbounds float, ptr %a2, i32 8
  store float 1.000000e+00, ptr %v139, align 4, !tbaa !0
  call void @llvm.lifetime.end.p0(i64 256, ptr %v0) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"any pointer", !2}
