; RUN: llc -O3 -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon-unknown--elf"

@g0 = external global void (float*, i32, i32, float*, float*)**

; Function Attrs: nounwind
define void @f0(float* nocapture %a0, float* nocapture %a1, float* %a2) #0 {
b0:
  %v0 = alloca [64 x float], align 16
  %v1 = alloca [8 x float], align 8
  %v2 = bitcast [64 x float]* %v0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 256, i8* %v2) #2
  %v3 = load float, float* %a0, align 4, !tbaa !0
  %v4 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 35
  store float %v3, float* %v4, align 4, !tbaa !0
  %v5 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 0
  store float %v3, float* %v5, align 16, !tbaa !0
  %v6 = getelementptr inbounds float, float* %a0, i32 1
  %v7 = load float, float* %v6, align 4, !tbaa !0
  %v8 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 36
  store float %v7, float* %v8, align 16, !tbaa !0
  %v9 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 1
  store float %v7, float* %v9, align 4, !tbaa !0
  %v10 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 37
  store float 1.000000e+00, float* %v10, align 4, !tbaa !0
  %v11 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 2
  store float 1.000000e+00, float* %v11, align 8, !tbaa !0
  %v12 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 34
  store float 0.000000e+00, float* %v12, align 8, !tbaa !0
  %v13 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 33
  store float 0.000000e+00, float* %v13, align 4, !tbaa !0
  %v14 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 32
  store float 0.000000e+00, float* %v14, align 16, !tbaa !0
  %v15 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 5
  store float 0.000000e+00, float* %v15, align 4, !tbaa !0
  %v16 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 4
  store float 0.000000e+00, float* %v16, align 16, !tbaa !0
  %v17 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 3
  store float 0.000000e+00, float* %v17, align 4, !tbaa !0
  %v18 = load float, float* %a1, align 4, !tbaa !0
  %v19 = fmul float %v3, %v18
  %v20 = fsub float -0.000000e+00, %v19
  %v21 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 6
  store float %v20, float* %v21, align 8, !tbaa !0
  %v22 = fmul float %v7, %v18
  %v23 = fsub float -0.000000e+00, %v22
  %v24 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 7
  store float %v23, float* %v24, align 4, !tbaa !0
  %v25 = getelementptr inbounds float, float* %a1, i32 1
  %v26 = load float, float* %v25, align 4, !tbaa !0
  %v27 = fmul float %v3, %v26
  %v28 = fsub float -0.000000e+00, %v27
  %v29 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 38
  store float %v28, float* %v29, align 8, !tbaa !0
  %v30 = fmul float %v7, %v26
  %v31 = fsub float -0.000000e+00, %v30
  %v32 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 39
  store float %v31, float* %v32, align 4, !tbaa !0
  %v33 = getelementptr inbounds [8 x float], [8 x float]* %v1, i32 0, i32 0
  store float %v18, float* %v33, align 8, !tbaa !0
  %v34 = getelementptr inbounds [8 x float], [8 x float]* %v1, i32 0, i32 4
  store float %v26, float* %v34, align 8, !tbaa !0
  %v35 = getelementptr float, float* %a0, i32 2
  %v36 = getelementptr float, float* %a1, i32 2
  %v37 = load float, float* %v35, align 4, !tbaa !0
  %v38 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 43
  store float %v37, float* %v38, align 4, !tbaa !0
  %v39 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 8
  store float %v37, float* %v39, align 16, !tbaa !0
  %v40 = getelementptr inbounds float, float* %a0, i32 3
  %v41 = load float, float* %v40, align 4, !tbaa !0
  %v42 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 44
  store float %v41, float* %v42, align 16, !tbaa !0
  %v43 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 9
  store float %v41, float* %v43, align 4, !tbaa !0
  %v44 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 45
  store float 1.000000e+00, float* %v44, align 4, !tbaa !0
  %v45 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 10
  store float 1.000000e+00, float* %v45, align 8, !tbaa !0
  %v46 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 42
  store float 0.000000e+00, float* %v46, align 8, !tbaa !0
  %v47 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 41
  store float 0.000000e+00, float* %v47, align 4, !tbaa !0
  %v48 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 40
  store float 0.000000e+00, float* %v48, align 16, !tbaa !0
  %v49 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 13
  store float 0.000000e+00, float* %v49, align 4, !tbaa !0
  %v50 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 12
  store float 0.000000e+00, float* %v50, align 16, !tbaa !0
  %v51 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 11
  store float 0.000000e+00, float* %v51, align 4, !tbaa !0
  %v52 = load float, float* %v36, align 4, !tbaa !0
  %v53 = fmul float %v37, %v52
  %v54 = fsub float -0.000000e+00, %v53
  %v55 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 14
  store float %v54, float* %v55, align 8, !tbaa !0
  %v56 = fmul float %v41, %v52
  %v57 = fsub float -0.000000e+00, %v56
  %v58 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 15
  store float %v57, float* %v58, align 4, !tbaa !0
  %v59 = getelementptr inbounds float, float* %a1, i32 3
  %v60 = load float, float* %v59, align 4, !tbaa !0
  %v61 = fmul float %v37, %v60
  %v62 = fsub float -0.000000e+00, %v61
  %v63 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 46
  store float %v62, float* %v63, align 8, !tbaa !0
  %v64 = fmul float %v41, %v60
  %v65 = fsub float -0.000000e+00, %v64
  %v66 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 47
  store float %v65, float* %v66, align 4, !tbaa !0
  %v67 = getelementptr inbounds [8 x float], [8 x float]* %v1, i32 0, i32 1
  store float %v52, float* %v67, align 4, !tbaa !0
  %v68 = getelementptr inbounds [8 x float], [8 x float]* %v1, i32 0, i32 5
  store float %v60, float* %v68, align 4, !tbaa !0
  %v69 = getelementptr float, float* %a0, i32 4
  %v70 = getelementptr float, float* %a1, i32 4
  %v71 = load float, float* %v69, align 4, !tbaa !0
  %v72 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 51
  store float %v71, float* %v72, align 4, !tbaa !0
  %v73 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 16
  store float %v71, float* %v73, align 16, !tbaa !0
  %v74 = getelementptr inbounds float, float* %a0, i32 5
  %v75 = load float, float* %v74, align 4, !tbaa !0
  %v76 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 52
  store float %v75, float* %v76, align 16, !tbaa !0
  %v77 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 17
  store float %v75, float* %v77, align 4, !tbaa !0
  %v78 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 53
  store float 1.000000e+00, float* %v78, align 4, !tbaa !0
  %v79 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 18
  store float 1.000000e+00, float* %v79, align 8, !tbaa !0
  %v80 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 50
  store float 0.000000e+00, float* %v80, align 8, !tbaa !0
  %v81 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 49
  store float 0.000000e+00, float* %v81, align 4, !tbaa !0
  %v82 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 48
  store float 0.000000e+00, float* %v82, align 16, !tbaa !0
  %v83 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 21
  store float 0.000000e+00, float* %v83, align 4, !tbaa !0
  %v84 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 20
  store float 0.000000e+00, float* %v84, align 16, !tbaa !0
  %v85 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 19
  store float 0.000000e+00, float* %v85, align 4, !tbaa !0
  %v86 = load float, float* %v70, align 4, !tbaa !0
  %v87 = fmul float %v71, %v86
  %v88 = fsub float -0.000000e+00, %v87
  %v89 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 22
  store float %v88, float* %v89, align 8, !tbaa !0
  %v90 = fmul float %v75, %v86
  %v91 = fsub float -0.000000e+00, %v90
  %v92 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 23
  store float %v91, float* %v92, align 4, !tbaa !0
  %v93 = getelementptr inbounds float, float* %a1, i32 5
  %v94 = load float, float* %v93, align 4, !tbaa !0
  %v95 = fmul float %v71, %v94
  %v96 = fsub float -0.000000e+00, %v95
  %v97 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 54
  store float %v96, float* %v97, align 8, !tbaa !0
  %v98 = fmul float %v75, %v94
  %v99 = fsub float -0.000000e+00, %v98
  %v100 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 55
  store float %v99, float* %v100, align 4, !tbaa !0
  %v101 = getelementptr inbounds [8 x float], [8 x float]* %v1, i32 0, i32 2
  store float %v86, float* %v101, align 8, !tbaa !0
  %v102 = getelementptr inbounds [8 x float], [8 x float]* %v1, i32 0, i32 6
  store float %v94, float* %v102, align 8, !tbaa !0
  %v103 = getelementptr float, float* %a0, i32 6
  %v104 = getelementptr float, float* %a1, i32 6
  %v105 = load float, float* %v103, align 4, !tbaa !0
  %v106 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 59
  store float %v105, float* %v106, align 4, !tbaa !0
  %v107 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 24
  store float %v105, float* %v107, align 16, !tbaa !0
  %v108 = getelementptr inbounds float, float* %a0, i32 7
  %v109 = load float, float* %v108, align 4, !tbaa !0
  %v110 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 60
  store float %v109, float* %v110, align 16, !tbaa !0
  %v111 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 25
  store float %v109, float* %v111, align 4, !tbaa !0
  %v112 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 61
  store float 1.000000e+00, float* %v112, align 4, !tbaa !0
  %v113 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 26
  store float 1.000000e+00, float* %v113, align 8, !tbaa !0
  %v114 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 58
  store float 0.000000e+00, float* %v114, align 8, !tbaa !0
  %v115 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 57
  store float 0.000000e+00, float* %v115, align 4, !tbaa !0
  %v116 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 56
  store float 0.000000e+00, float* %v116, align 16, !tbaa !0
  %v117 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 29
  store float 0.000000e+00, float* %v117, align 4, !tbaa !0
  %v118 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 28
  store float 0.000000e+00, float* %v118, align 16, !tbaa !0
  %v119 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 27
  store float 0.000000e+00, float* %v119, align 4, !tbaa !0
  %v120 = load float, float* %v104, align 4, !tbaa !0
  %v121 = fmul float %v105, %v120
  %v122 = fsub float -0.000000e+00, %v121
  %v123 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 30
  store float %v122, float* %v123, align 8, !tbaa !0
  %v124 = fmul float %v109, %v120
  %v125 = fsub float -0.000000e+00, %v124
  %v126 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 31
  store float %v125, float* %v126, align 4, !tbaa !0
  %v127 = getelementptr inbounds float, float* %a1, i32 7
  %v128 = load float, float* %v127, align 4, !tbaa !0
  %v129 = fmul float %v105, %v128
  %v130 = fsub float -0.000000e+00, %v129
  %v131 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 62
  store float %v130, float* %v131, align 8, !tbaa !0
  %v132 = fmul float %v109, %v128
  %v133 = fsub float -0.000000e+00, %v132
  %v134 = getelementptr inbounds [64 x float], [64 x float]* %v0, i32 0, i32 63
  store float %v133, float* %v134, align 4, !tbaa !0
  %v135 = getelementptr inbounds [8 x float], [8 x float]* %v1, i32 0, i32 3
  store float %v120, float* %v135, align 4, !tbaa !0
  %v136 = getelementptr inbounds [8 x float], [8 x float]* %v1, i32 0, i32 7
  store float %v128, float* %v136, align 4, !tbaa !0
  %v137 = load void (float*, i32, i32, float*, float*)**, void (float*, i32, i32, float*, float*)*** @g0, align 4, !tbaa !4
  %v138 = load void (float*, i32, i32, float*, float*)*, void (float*, i32, i32, float*, float*)** %v137, align 4, !tbaa !4
  call void %v138(float* %v5, i32 8, i32 8, float* %v33, float* %a2) #2
  %v139 = getelementptr inbounds float, float* %a2, i32 8
  store float 1.000000e+00, float* %v139, align 4, !tbaa !0
  call void @llvm.lifetime.end.p0i8(i64 256, i8* %v2) #2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"any pointer", !2}
