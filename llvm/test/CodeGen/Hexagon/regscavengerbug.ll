; RUN: llc -march=hexagon -O3 < %s
; REQUIRES: asserts

; This used to assert in the register scavenger.

target triple = "hexagon-unknown-linux-gnu"

%0 = type { %1 }
%1 = type { %2 }
%2 = type { [4 x [4 x double]] }
%3 = type { [3 x double] }
%4 = type { %5, %0, %0, %5*, %3, %3 }
%5 = type { i32 (...)** }
%6 = type { %3, %3 }

declare void @f0(%3* sret(%3), %0*, %3*)

; Function Attrs: nounwind
define void @f1(%4* %a0, %0* nocapture %a1, %0* nocapture %a2) #0 align 2 {
b0:
  %v0 = alloca %6, align 8
  %v1 = alloca [2 x [2 x [2 x %3]]], align 8
  %v2 = alloca %3, align 8
  %v3 = getelementptr inbounds %4, %4* %a0, i32 0, i32 1
  %v4 = bitcast %0* %v3 to i8*
  %v5 = bitcast %0* %a1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v4, i8* align 8 %v5, i32 128, i1 false)
  %v6 = getelementptr inbounds %4, %4* %a0, i32 0, i32 2
  %v7 = bitcast %0* %v6 to i8*
  %v8 = bitcast %0* %a2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v7, i8* align 8 %v8, i32 128, i1 false)
  %v9 = bitcast %6* %v0 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %v9, i8 0, i64 48, i1 false)
  %v10 = getelementptr inbounds %4, %4* %a0, i32 0, i32 3
  %v11 = load %5*, %5** %v10, align 4, !tbaa !0
  %v12 = bitcast %5* %v11 to i32 (%5*, double, double, %6*)***
  %v13 = load i32 (%5*, double, double, %6*)**, i32 (%5*, double, double, %6*)*** %v12, align 4, !tbaa !4
  %v14 = getelementptr inbounds i32 (%5*, double, double, %6*)*, i32 (%5*, double, double, %6*)** %v13, i32 3
  %v15 = load i32 (%5*, double, double, %6*)*, i32 (%5*, double, double, %6*)** %v14, align 4
  %v16 = call i32 %v15(%5* %v11, double 0.000000e+00, double 0.000000e+00, %6* %v0)
  %v17 = icmp eq i32 %v16, 0
  br i1 %v17, label %b1, label %b3

b1:                                               ; preds = %b0
  %v18 = getelementptr inbounds %4, %4* %a0, i32 0, i32 4, i32 0, i32 0
  store double -1.000000e+06, double* %v18, align 8, !tbaa !6
  %v19 = getelementptr inbounds %4, %4* %a0, i32 0, i32 4, i32 0, i32 1
  store double -1.000000e+06, double* %v19, align 8, !tbaa !6
  %v20 = getelementptr inbounds %4, %4* %a0, i32 0, i32 4, i32 0, i32 2
  store double -1.000000e+06, double* %v20, align 8, !tbaa !6
  %v21 = getelementptr inbounds %4, %4* %a0, i32 0, i32 5, i32 0, i32 0
  store double 1.000000e+06, double* %v21, align 8, !tbaa !6
  %v22 = getelementptr inbounds %4, %4* %a0, i32 0, i32 5, i32 0, i32 1
  store double 1.000000e+06, double* %v22, align 8, !tbaa !6
  %v23 = getelementptr inbounds %4, %4* %a0, i32 0, i32 5, i32 0, i32 2
  store double 1.000000e+06, double* %v23, align 8, !tbaa !6
  br label %b2

b2:                                               ; preds = %b3, %b1
  ret void

b3:                                               ; preds = %b0
  %v24 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 0
  %v25 = bitcast [2 x [2 x [2 x %3]]]* %v1 to i8*
  %v26 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 2
  %v27 = bitcast %3* %v26 to i8*
  %v28 = bitcast [2 x [2 x [2 x %3]]]* %v1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %v28, i8 0, i64 48, i1 false)
  call void @llvm.memset.p0i8.i64(i8* align 8 %v27, i8 0, i64 24, i1 false)
  %v29 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 3
  %v30 = bitcast %3* %v29 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %v30, i8 0, i64 24, i1 false)
  %v31 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 4
  %v32 = bitcast %3* %v31 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %v32, i8 0, i64 24, i1 false)
  %v33 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 5
  %v34 = bitcast %3* %v33 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %v34, i8 0, i64 24, i1 false)
  %v35 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 6
  %v36 = bitcast %3* %v35 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %v36, i8 0, i64 24, i1 false)
  %v37 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 7
  %v38 = bitcast %3* %v37 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %v38, i8 0, i64 24, i1 false)
  %v39 = getelementptr inbounds %6, %6* %v0, i32 0, i32 0, i32 0, i32 0
  %v40 = getelementptr inbounds %6, %6* %v0, i32 0, i32 0, i32 0, i32 1
  %v41 = getelementptr inbounds %6, %6* %v0, i32 0, i32 0, i32 0, i32 2
  %v42 = bitcast %3* %v2 to i8*
  %v43 = getelementptr inbounds %6, %6* %v0, i32 0, i32 1, i32 0, i32 2
  %v44 = getelementptr inbounds %6, %6* %v0, i32 0, i32 1, i32 0, i32 1
  %v45 = getelementptr inbounds %6, %6* %v0, i32 0, i32 1, i32 0, i32 0
  %v46 = load double, double* %v39, align 8, !tbaa !6
  %v47 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store double %v46, double* %v47, align 8, !tbaa !6
  %v48 = load double, double* %v40, align 8, !tbaa !6
  %v49 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  store double %v48, double* %v49, align 8, !tbaa !6
  %v50 = load double, double* %v41, align 8, !tbaa !6
  %v51 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2
  store double %v50, double* %v51, align 8, !tbaa !6
  call void @f0(%3* sret(%3) %v2, %0* %v3, %3* %v24)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v25, i8* align 8 %v42, i32 24, i1 false)
  %v52 = load double, double* %v39, align 8, !tbaa !6
  %v53 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0
  store double %v52, double* %v53, align 8, !tbaa !6
  %v54 = load double, double* %v40, align 8, !tbaa !6
  %v55 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1
  store double %v54, double* %v55, align 8, !tbaa !6
  %v56 = load double, double* %v43, align 8, !tbaa !6
  %v57 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 2
  store double %v56, double* %v57, align 8, !tbaa !6
  %v58 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 1
  call void @f0(%3* sret(%3) %v2, %0* %v3, %3* %v58)
  %v59 = bitcast %3* %v58 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v59, i8* align 8 %v42, i32 24, i1 false)
  %v60 = load double, double* %v39, align 8, !tbaa !6
  %v61 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  store double %v60, double* %v61, align 8, !tbaa !6
  %v62 = load double, double* %v44, align 8, !tbaa !6
  %v63 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1
  store double %v62, double* %v63, align 8, !tbaa !6
  %v64 = load double, double* %v41, align 8, !tbaa !6
  %v65 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 2
  store double %v64, double* %v65, align 8, !tbaa !6
  %v66 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 0
  call void @f0(%3* sret(%3) %v2, %0* %v3, %3* %v66)
  %v67 = bitcast %3* %v66 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v67, i8* align 8 %v42, i32 24, i1 false)
  %v68 = load double, double* %v39, align 8, !tbaa !6
  %v69 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0
  store double %v68, double* %v69, align 8, !tbaa !6
  %v70 = load double, double* %v44, align 8, !tbaa !6
  %v71 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1
  store double %v70, double* %v71, align 8, !tbaa !6
  %v72 = load double, double* %v43, align 8, !tbaa !6
  %v73 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 2
  store double %v72, double* %v73, align 8, !tbaa !6
  %v74 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 1
  call void @f0(%3* sret(%3) %v2, %0* %v3, %3* %v74)
  %v75 = bitcast %3* %v74 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v75, i8* align 8 %v42, i32 24, i1 false)
  %v76 = load double, double* %v45, align 8, !tbaa !6
  %v77 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0
  store double %v76, double* %v77, align 8, !tbaa !6
  %v78 = load double, double* %v40, align 8, !tbaa !6
  %v79 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 1
  store double %v78, double* %v79, align 8, !tbaa !6
  %v80 = load double, double* %v41, align 8, !tbaa !6
  %v81 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 2
  store double %v80, double* %v81, align 8, !tbaa !6
  %v82 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 0
  call void @f0(%3* sret(%3) %v2, %0* %v3, %3* %v82)
  %v83 = bitcast %3* %v82 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v83, i8* align 8 %v42, i32 24, i1 false)
  %v84 = load double, double* %v45, align 8, !tbaa !6
  %v85 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0
  store double %v84, double* %v85, align 8, !tbaa !6
  %v86 = load double, double* %v40, align 8, !tbaa !6
  %v87 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1
  store double %v86, double* %v87, align 8, !tbaa !6
  %v88 = load double, double* %v43, align 8, !tbaa !6
  %v89 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 2
  store double %v88, double* %v89, align 8, !tbaa !6
  %v90 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 1
  call void @f0(%3* sret(%3) %v2, %0* %v3, %3* %v90)
  %v91 = bitcast %3* %v90 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v91, i8* align 8 %v42, i32 24, i1 false)
  %v92 = load double, double* %v45, align 8, !tbaa !6
  %v93 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0
  store double %v92, double* %v93, align 8, !tbaa !6
  %v94 = load double, double* %v44, align 8, !tbaa !6
  %v95 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 1
  store double %v94, double* %v95, align 8, !tbaa !6
  %v96 = load double, double* %v41, align 8, !tbaa !6
  %v97 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 2
  store double %v96, double* %v97, align 8, !tbaa !6
  %v98 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 0
  call void @f0(%3* sret(%3) %v2, %0* %v3, %3* %v98)
  %v99 = bitcast %3* %v98 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v99, i8* align 8 %v42, i32 24, i1 false)
  %v100 = load double, double* %v45, align 8, !tbaa !6
  %v101 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0
  store double %v100, double* %v101, align 8, !tbaa !6
  %v102 = load double, double* %v44, align 8, !tbaa !6
  %v103 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 1
  store double %v102, double* %v103, align 8, !tbaa !6
  %v104 = load double, double* %v43, align 8, !tbaa !6
  %v105 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 2
  store double %v104, double* %v105, align 8, !tbaa !6
  %v106 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 1
  call void @f0(%3* sret(%3) %v2, %0* %v3, %3* %v106)
  %v107 = bitcast %3* %v106 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v107, i8* align 8 %v42, i32 24, i1 false)
  %v108 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %v109 = load double, double* %v108, align 8, !tbaa !6
  %v110 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %v111 = load double, double* %v110, align 8, !tbaa !6
  %v112 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %v113 = load double, double* %v112, align 8, !tbaa !6
  %v114 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %v115 = load double, double* %v114, align 8, !tbaa !6
  %v116 = fcmp olt double %v115, %v109
  %v117 = select i1 %v116, double %v115, double %v109
  %v118 = fcmp ogt double %v115, %v109
  %v119 = select i1 %v118, double %v115, double %v109
  %v120 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1
  %v121 = load double, double* %v120, align 8, !tbaa !6
  %v122 = fcmp olt double %v121, %v111
  %v123 = select i1 %v122, double %v121, double %v111
  %v124 = fcmp ogt double %v121, %v111
  %v125 = select i1 %v124, double %v121, double %v111
  %v126 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 2
  %v127 = load double, double* %v126, align 8, !tbaa !6
  %v128 = fcmp olt double %v127, %v113
  %v129 = select i1 %v128, double %v127, double %v113
  %v130 = fcmp ogt double %v127, %v113
  %v131 = select i1 %v130, double %v127, double %v113
  %v132 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %v133 = load double, double* %v132, align 8, !tbaa !6
  %v134 = fcmp olt double %v133, %v117
  %v135 = select i1 %v134, double %v133, double %v117
  %v136 = fcmp ogt double %v133, %v119
  %v137 = select i1 %v136, double %v133, double %v119
  %v138 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1
  %v139 = load double, double* %v138, align 8, !tbaa !6
  %v140 = fcmp olt double %v139, %v123
  %v141 = select i1 %v140, double %v139, double %v123
  %v142 = fcmp ogt double %v139, %v125
  %v143 = select i1 %v142, double %v139, double %v125
  %v144 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 2
  %v145 = load double, double* %v144, align 8, !tbaa !6
  %v146 = fcmp olt double %v145, %v129
  %v147 = select i1 %v146, double %v145, double %v129
  %v148 = fcmp ogt double %v145, %v131
  %v149 = select i1 %v148, double %v145, double %v131
  %v150 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0
  %v151 = load double, double* %v150, align 8, !tbaa !6
  %v152 = fcmp olt double %v151, %v135
  %v153 = select i1 %v152, double %v151, double %v135
  %v154 = fcmp ogt double %v151, %v137
  %v155 = select i1 %v154, double %v151, double %v137
  %v156 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1
  %v157 = load double, double* %v156, align 8, !tbaa !6
  %v158 = fcmp olt double %v157, %v141
  %v159 = select i1 %v158, double %v157, double %v141
  %v160 = fcmp ogt double %v157, %v143
  %v161 = select i1 %v160, double %v157, double %v143
  %v162 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 2
  %v163 = load double, double* %v162, align 8, !tbaa !6
  %v164 = fcmp olt double %v163, %v147
  %v165 = select i1 %v164, double %v163, double %v147
  %v166 = fcmp ogt double %v163, %v149
  %v167 = select i1 %v166, double %v163, double %v149
  %v168 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0
  %v169 = load double, double* %v168, align 8, !tbaa !6
  %v170 = fcmp olt double %v169, %v153
  %v171 = select i1 %v170, double %v169, double %v153
  %v172 = fcmp ogt double %v169, %v155
  %v173 = select i1 %v172, double %v169, double %v155
  %v174 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 1
  %v175 = load double, double* %v174, align 8, !tbaa !6
  %v176 = fcmp olt double %v175, %v159
  %v177 = select i1 %v176, double %v175, double %v159
  %v178 = fcmp ogt double %v175, %v161
  %v179 = select i1 %v178, double %v175, double %v161
  %v180 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 2
  %v181 = load double, double* %v180, align 8, !tbaa !6
  %v182 = fcmp olt double %v181, %v165
  %v183 = select i1 %v182, double %v181, double %v165
  %v184 = fcmp ogt double %v181, %v167
  %v185 = select i1 %v184, double %v181, double %v167
  %v186 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0
  %v187 = load double, double* %v186, align 8, !tbaa !6
  %v188 = fcmp olt double %v187, %v171
  %v189 = select i1 %v188, double %v187, double %v171
  %v190 = fcmp ogt double %v187, %v173
  %v191 = select i1 %v190, double %v187, double %v173
  %v192 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1
  %v193 = load double, double* %v192, align 8, !tbaa !6
  %v194 = fcmp olt double %v193, %v177
  %v195 = select i1 %v194, double %v193, double %v177
  %v196 = fcmp ogt double %v193, %v179
  %v197 = select i1 %v196, double %v193, double %v179
  %v198 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 2
  %v199 = load double, double* %v198, align 8, !tbaa !6
  %v200 = fcmp olt double %v199, %v183
  %v201 = select i1 %v200, double %v199, double %v183
  %v202 = fcmp ogt double %v199, %v185
  %v203 = select i1 %v202, double %v199, double %v185
  %v204 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0
  %v205 = load double, double* %v204, align 8, !tbaa !6
  %v206 = fcmp olt double %v205, %v189
  %v207 = select i1 %v206, double %v205, double %v189
  %v208 = fcmp ogt double %v205, %v191
  %v209 = select i1 %v208, double %v205, double %v191
  %v210 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 1
  %v211 = load double, double* %v210, align 8, !tbaa !6
  %v212 = fcmp olt double %v211, %v195
  %v213 = select i1 %v212, double %v211, double %v195
  %v214 = fcmp ogt double %v211, %v197
  %v215 = select i1 %v214, double %v211, double %v197
  %v216 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 2
  %v217 = load double, double* %v216, align 8, !tbaa !6
  %v218 = fcmp olt double %v217, %v201
  %v219 = select i1 %v218, double %v217, double %v201
  %v220 = fcmp ogt double %v217, %v203
  %v221 = select i1 %v220, double %v217, double %v203
  %v222 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0
  %v223 = load double, double* %v222, align 8, !tbaa !6
  %v224 = fcmp olt double %v223, %v207
  %v225 = select i1 %v224, double %v223, double %v207
  %v226 = fcmp ogt double %v223, %v209
  %v227 = select i1 %v226, double %v223, double %v209
  %v228 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 1
  %v229 = load double, double* %v228, align 8, !tbaa !6
  %v230 = fcmp olt double %v229, %v213
  %v231 = select i1 %v230, double %v229, double %v213
  %v232 = fcmp ogt double %v229, %v215
  %v233 = select i1 %v232, double %v229, double %v215
  %v234 = getelementptr inbounds [2 x [2 x [2 x %3]]], [2 x [2 x [2 x %3]]]* %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 2
  %v235 = load double, double* %v234, align 8, !tbaa !6
  %v236 = fcmp olt double %v235, %v219
  %v237 = select i1 %v236, double %v235, double %v219
  %v238 = fcmp ogt double %v235, %v221
  %v239 = select i1 %v238, double %v235, double %v221
  %v240 = getelementptr inbounds %4, %4* %a0, i32 0, i32 4, i32 0, i32 0
  store double %v225, double* %v240, align 8
  %v241 = getelementptr inbounds %4, %4* %a0, i32 0, i32 4, i32 0, i32 1
  store double %v231, double* %v241, align 8
  %v242 = getelementptr inbounds %4, %4* %a0, i32 0, i32 4, i32 0, i32 2
  store double %v237, double* %v242, align 8
  %v243 = getelementptr inbounds %4, %4* %a0, i32 0, i32 5, i32 0, i32 0
  store double %v227, double* %v243, align 8
  %v244 = getelementptr inbounds %4, %4* %a0, i32 0, i32 5, i32 0, i32 1
  store double %v233, double* %v244, align 8
  %v245 = getelementptr inbounds %4, %4* %a0, i32 0, i32 5, i32 0, i32 2
  store double %v239, double* %v245, align 8
  br label %b2
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"vtable pointer", !3}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !2}
