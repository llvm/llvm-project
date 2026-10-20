; RUN: llc -mtriple=hexagon -O3 < %s
; REQUIRES: asserts

; This used to assert in the register scavenger.

target triple = "hexagon-unknown-linux-gnu"

%0 = type { %1 }
%1 = type { %2 }
%2 = type { [4 x [4 x double]] }
%3 = type { [3 x double] }
%4 = type { %5, %0, %0, ptr, %3, %3 }
%5 = type { ptr }
%6 = type { %3, %3 }

declare void @f0(ptr sret(%3), ptr, ptr)

; Function Attrs: nounwind
define void @f1(ptr %a0, ptr nocapture %a1, ptr nocapture %a2) #0 align 2 {
b0:
  %v0 = alloca %6, align 8
  %v1 = alloca [2 x [2 x [2 x %3]]], align 8
  %v2 = alloca %3, align 8
  %v3 = getelementptr inbounds %4, ptr %a0, i32 0, i32 1
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v3, ptr align 8 %a1, i32 128, i1 false)
  %v6 = getelementptr inbounds %4, ptr %a0, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v6, ptr align 8 %a2, i32 128, i1 false)
  call void @llvm.memset.p0.i64(ptr align 8 %v0, i8 0, i64 48, i1 false)
  %v10 = getelementptr inbounds %4, ptr %a0, i32 0, i32 3
  %v11 = load ptr, ptr %v10, align 4, !tbaa !0
  %v13 = load ptr, ptr %v11, align 4, !tbaa !4
  %v14 = getelementptr inbounds ptr, ptr %v13, i32 3
  %v15 = load ptr, ptr %v14, align 4
  %v16 = call i32 %v15(ptr %v11, double 0.000000e+00, double 0.000000e+00, ptr %v0)
  %v17 = icmp eq i32 %v16, 0
  br i1 %v17, label %b1, label %b3

b1:                                               ; preds = %b0
  %v18 = getelementptr inbounds %4, ptr %a0, i32 0, i32 4, i32 0, i32 0
  store double -1.000000e+06, ptr %v18, align 8, !tbaa !6
  %v19 = getelementptr inbounds %4, ptr %a0, i32 0, i32 4, i32 0, i32 1
  store double -1.000000e+06, ptr %v19, align 8, !tbaa !6
  %v20 = getelementptr inbounds %4, ptr %a0, i32 0, i32 4, i32 0, i32 2
  store double -1.000000e+06, ptr %v20, align 8, !tbaa !6
  %v21 = getelementptr inbounds %4, ptr %a0, i32 0, i32 5, i32 0, i32 0
  store double 1.000000e+06, ptr %v21, align 8, !tbaa !6
  %v22 = getelementptr inbounds %4, ptr %a0, i32 0, i32 5, i32 0, i32 1
  store double 1.000000e+06, ptr %v22, align 8, !tbaa !6
  %v23 = getelementptr inbounds %4, ptr %a0, i32 0, i32 5, i32 0, i32 2
  store double 1.000000e+06, ptr %v23, align 8, !tbaa !6
  br label %b2

b2:                                               ; preds = %b3, %b1
  ret void

b3:                                               ; preds = %b0
  %v26 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 2
  call void @llvm.memset.p0.i64(ptr align 8 %v1, i8 0, i64 48, i1 false)
  call void @llvm.memset.p0.i64(ptr align 8 %v26, i8 0, i64 24, i1 false)
  %v29 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 3
  call void @llvm.memset.p0.i64(ptr align 8 %v29, i8 0, i64 24, i1 false)
  %v31 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 4
  call void @llvm.memset.p0.i64(ptr align 8 %v31, i8 0, i64 24, i1 false)
  %v33 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 5
  call void @llvm.memset.p0.i64(ptr align 8 %v33, i8 0, i64 24, i1 false)
  %v35 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 6
  call void @llvm.memset.p0.i64(ptr align 8 %v35, i8 0, i64 24, i1 false)
  %v37 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 7
  call void @llvm.memset.p0.i64(ptr align 8 %v37, i8 0, i64 24, i1 false)
  %v40 = getelementptr inbounds %6, ptr %v0, i32 0, i32 0, i32 0, i32 1
  %v41 = getelementptr inbounds %6, ptr %v0, i32 0, i32 0, i32 0, i32 2
  %v43 = getelementptr inbounds %6, ptr %v0, i32 0, i32 1, i32 0, i32 2
  %v44 = getelementptr inbounds %6, ptr %v0, i32 0, i32 1, i32 0, i32 1
  %v45 = getelementptr inbounds %6, ptr %v0, i32 0, i32 1, i32 0, i32 0
  %v46 = load double, ptr %v0, align 8, !tbaa !6
  store double %v46, ptr %v1, align 8, !tbaa !6
  %v48 = load double, ptr %v40, align 8, !tbaa !6
  %v49 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  store double %v48, ptr %v49, align 8, !tbaa !6
  %v50 = load double, ptr %v41, align 8, !tbaa !6
  %v51 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2
  store double %v50, ptr %v51, align 8, !tbaa !6
  call void @f0(ptr sret(%3) %v2, ptr %v3, ptr %v1)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v1, ptr align 8 %v2, i32 24, i1 false)
  %v52 = load double, ptr %v0, align 8, !tbaa !6
  %v53 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0
  store double %v52, ptr %v53, align 8, !tbaa !6
  %v54 = load double, ptr %v40, align 8, !tbaa !6
  %v55 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1
  store double %v54, ptr %v55, align 8, !tbaa !6
  %v56 = load double, ptr %v43, align 8, !tbaa !6
  %v57 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 2
  store double %v56, ptr %v57, align 8, !tbaa !6
  %v58 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 1
  call void @f0(ptr sret(%3) %v2, ptr %v3, ptr %v58)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v58, ptr align 8 %v2, i32 24, i1 false)
  %v60 = load double, ptr %v0, align 8, !tbaa !6
  %v61 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  store double %v60, ptr %v61, align 8, !tbaa !6
  %v62 = load double, ptr %v44, align 8, !tbaa !6
  %v63 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1
  store double %v62, ptr %v63, align 8, !tbaa !6
  %v64 = load double, ptr %v41, align 8, !tbaa !6
  %v65 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 2
  store double %v64, ptr %v65, align 8, !tbaa !6
  %v66 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 0
  call void @f0(ptr sret(%3) %v2, ptr %v3, ptr %v66)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v66, ptr align 8 %v2, i32 24, i1 false)
  %v68 = load double, ptr %v0, align 8, !tbaa !6
  %v69 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0
  store double %v68, ptr %v69, align 8, !tbaa !6
  %v70 = load double, ptr %v44, align 8, !tbaa !6
  %v71 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1
  store double %v70, ptr %v71, align 8, !tbaa !6
  %v72 = load double, ptr %v43, align 8, !tbaa !6
  %v73 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 2
  store double %v72, ptr %v73, align 8, !tbaa !6
  %v74 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 1
  call void @f0(ptr sret(%3) %v2, ptr %v3, ptr %v74)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v74, ptr align 8 %v2, i32 24, i1 false)
  %v76 = load double, ptr %v45, align 8, !tbaa !6
  %v77 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0
  store double %v76, ptr %v77, align 8, !tbaa !6
  %v78 = load double, ptr %v40, align 8, !tbaa !6
  %v79 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 1
  store double %v78, ptr %v79, align 8, !tbaa !6
  %v80 = load double, ptr %v41, align 8, !tbaa !6
  %v81 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 2
  store double %v80, ptr %v81, align 8, !tbaa !6
  %v82 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 0
  call void @f0(ptr sret(%3) %v2, ptr %v3, ptr %v82)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v82, ptr align 8 %v2, i32 24, i1 false)
  %v84 = load double, ptr %v45, align 8, !tbaa !6
  %v85 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0
  store double %v84, ptr %v85, align 8, !tbaa !6
  %v86 = load double, ptr %v40, align 8, !tbaa !6
  %v87 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1
  store double %v86, ptr %v87, align 8, !tbaa !6
  %v88 = load double, ptr %v43, align 8, !tbaa !6
  %v89 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 2
  store double %v88, ptr %v89, align 8, !tbaa !6
  %v90 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 1
  call void @f0(ptr sret(%3) %v2, ptr %v3, ptr %v90)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v90, ptr align 8 %v2, i32 24, i1 false)
  %v92 = load double, ptr %v45, align 8, !tbaa !6
  %v93 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0
  store double %v92, ptr %v93, align 8, !tbaa !6
  %v94 = load double, ptr %v44, align 8, !tbaa !6
  %v95 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 1
  store double %v94, ptr %v95, align 8, !tbaa !6
  %v96 = load double, ptr %v41, align 8, !tbaa !6
  %v97 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 2
  store double %v96, ptr %v97, align 8, !tbaa !6
  %v98 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 0
  call void @f0(ptr sret(%3) %v2, ptr %v3, ptr %v98)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v98, ptr align 8 %v2, i32 24, i1 false)
  %v100 = load double, ptr %v45, align 8, !tbaa !6
  %v101 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0
  store double %v100, ptr %v101, align 8, !tbaa !6
  %v102 = load double, ptr %v44, align 8, !tbaa !6
  %v103 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 1
  store double %v102, ptr %v103, align 8, !tbaa !6
  %v104 = load double, ptr %v43, align 8, !tbaa !6
  %v105 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 2
  store double %v104, ptr %v105, align 8, !tbaa !6
  %v106 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 1
  call void @f0(ptr sret(%3) %v2, ptr %v3, ptr %v106)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %v106, ptr align 8 %v2, i32 24, i1 false)
  %v109 = load double, ptr %v1, align 8, !tbaa !6
  %v110 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1
  %v111 = load double, ptr %v110, align 8, !tbaa !6
  %v112 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %v113 = load double, ptr %v112, align 8, !tbaa !6
  %v114 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0
  %v115 = load double, ptr %v114, align 8, !tbaa !6
  %v116 = fcmp olt double %v115, %v109
  %v117 = select i1 %v116, double %v115, double %v109
  %v118 = fcmp ogt double %v115, %v109
  %v119 = select i1 %v118, double %v115, double %v109
  %v120 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1
  %v121 = load double, ptr %v120, align 8, !tbaa !6
  %v122 = fcmp olt double %v121, %v111
  %v123 = select i1 %v122, double %v121, double %v111
  %v124 = fcmp ogt double %v121, %v111
  %v125 = select i1 %v124, double %v121, double %v111
  %v126 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 2
  %v127 = load double, ptr %v126, align 8, !tbaa !6
  %v128 = fcmp olt double %v127, %v113
  %v129 = select i1 %v128, double %v127, double %v113
  %v130 = fcmp ogt double %v127, %v113
  %v131 = select i1 %v130, double %v127, double %v113
  %v132 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %v133 = load double, ptr %v132, align 8, !tbaa !6
  %v134 = fcmp olt double %v133, %v117
  %v135 = select i1 %v134, double %v133, double %v117
  %v136 = fcmp ogt double %v133, %v119
  %v137 = select i1 %v136, double %v133, double %v119
  %v138 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1
  %v139 = load double, ptr %v138, align 8, !tbaa !6
  %v140 = fcmp olt double %v139, %v123
  %v141 = select i1 %v140, double %v139, double %v123
  %v142 = fcmp ogt double %v139, %v125
  %v143 = select i1 %v142, double %v139, double %v125
  %v144 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 2
  %v145 = load double, ptr %v144, align 8, !tbaa !6
  %v146 = fcmp olt double %v145, %v129
  %v147 = select i1 %v146, double %v145, double %v129
  %v148 = fcmp ogt double %v145, %v131
  %v149 = select i1 %v148, double %v145, double %v131
  %v150 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0
  %v151 = load double, ptr %v150, align 8, !tbaa !6
  %v152 = fcmp olt double %v151, %v135
  %v153 = select i1 %v152, double %v151, double %v135
  %v154 = fcmp ogt double %v151, %v137
  %v155 = select i1 %v154, double %v151, double %v137
  %v156 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1
  %v157 = load double, ptr %v156, align 8, !tbaa !6
  %v158 = fcmp olt double %v157, %v141
  %v159 = select i1 %v158, double %v157, double %v141
  %v160 = fcmp ogt double %v157, %v143
  %v161 = select i1 %v160, double %v157, double %v143
  %v162 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 0, i32 1, i32 1, i32 0, i32 2
  %v163 = load double, ptr %v162, align 8, !tbaa !6
  %v164 = fcmp olt double %v163, %v147
  %v165 = select i1 %v164, double %v163, double %v147
  %v166 = fcmp ogt double %v163, %v149
  %v167 = select i1 %v166, double %v163, double %v149
  %v168 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0
  %v169 = load double, ptr %v168, align 8, !tbaa !6
  %v170 = fcmp olt double %v169, %v153
  %v171 = select i1 %v170, double %v169, double %v153
  %v172 = fcmp ogt double %v169, %v155
  %v173 = select i1 %v172, double %v169, double %v155
  %v174 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 1
  %v175 = load double, ptr %v174, align 8, !tbaa !6
  %v176 = fcmp olt double %v175, %v159
  %v177 = select i1 %v176, double %v175, double %v159
  %v178 = fcmp ogt double %v175, %v161
  %v179 = select i1 %v178, double %v175, double %v161
  %v180 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 0, i32 0, i32 2
  %v181 = load double, ptr %v180, align 8, !tbaa !6
  %v182 = fcmp olt double %v181, %v165
  %v183 = select i1 %v182, double %v181, double %v165
  %v184 = fcmp ogt double %v181, %v167
  %v185 = select i1 %v184, double %v181, double %v167
  %v186 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0
  %v187 = load double, ptr %v186, align 8, !tbaa !6
  %v188 = fcmp olt double %v187, %v171
  %v189 = select i1 %v188, double %v187, double %v171
  %v190 = fcmp ogt double %v187, %v173
  %v191 = select i1 %v190, double %v187, double %v173
  %v192 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1
  %v193 = load double, ptr %v192, align 8, !tbaa !6
  %v194 = fcmp olt double %v193, %v177
  %v195 = select i1 %v194, double %v193, double %v177
  %v196 = fcmp ogt double %v193, %v179
  %v197 = select i1 %v196, double %v193, double %v179
  %v198 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 2
  %v199 = load double, ptr %v198, align 8, !tbaa !6
  %v200 = fcmp olt double %v199, %v183
  %v201 = select i1 %v200, double %v199, double %v183
  %v202 = fcmp ogt double %v199, %v185
  %v203 = select i1 %v202, double %v199, double %v185
  %v204 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0
  %v205 = load double, ptr %v204, align 8, !tbaa !6
  %v206 = fcmp olt double %v205, %v189
  %v207 = select i1 %v206, double %v205, double %v189
  %v208 = fcmp ogt double %v205, %v191
  %v209 = select i1 %v208, double %v205, double %v191
  %v210 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 1
  %v211 = load double, ptr %v210, align 8, !tbaa !6
  %v212 = fcmp olt double %v211, %v195
  %v213 = select i1 %v212, double %v211, double %v195
  %v214 = fcmp ogt double %v211, %v197
  %v215 = select i1 %v214, double %v211, double %v197
  %v216 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 0, i32 0, i32 2
  %v217 = load double, ptr %v216, align 8, !tbaa !6
  %v218 = fcmp olt double %v217, %v201
  %v219 = select i1 %v218, double %v217, double %v201
  %v220 = fcmp ogt double %v217, %v203
  %v221 = select i1 %v220, double %v217, double %v203
  %v222 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 0
  %v223 = load double, ptr %v222, align 8, !tbaa !6
  %v224 = fcmp olt double %v223, %v207
  %v225 = select i1 %v224, double %v223, double %v207
  %v226 = fcmp ogt double %v223, %v209
  %v227 = select i1 %v226, double %v223, double %v209
  %v228 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 1
  %v229 = load double, ptr %v228, align 8, !tbaa !6
  %v230 = fcmp olt double %v229, %v213
  %v231 = select i1 %v230, double %v229, double %v213
  %v232 = fcmp ogt double %v229, %v215
  %v233 = select i1 %v232, double %v229, double %v215
  %v234 = getelementptr inbounds [2 x [2 x [2 x %3]]], ptr %v1, i32 0, i32 1, i32 1, i32 1, i32 0, i32 2
  %v235 = load double, ptr %v234, align 8, !tbaa !6
  %v236 = fcmp olt double %v235, %v219
  %v237 = select i1 %v236, double %v235, double %v219
  %v238 = fcmp ogt double %v235, %v221
  %v239 = select i1 %v238, double %v235, double %v221
  %v240 = getelementptr inbounds %4, ptr %a0, i32 0, i32 4, i32 0, i32 0
  store double %v225, ptr %v240, align 8
  %v241 = getelementptr inbounds %4, ptr %a0, i32 0, i32 4, i32 0, i32 1
  store double %v231, ptr %v241, align 8
  %v242 = getelementptr inbounds %4, ptr %a0, i32 0, i32 4, i32 0, i32 2
  store double %v237, ptr %v242, align 8
  %v243 = getelementptr inbounds %4, ptr %a0, i32 0, i32 5, i32 0, i32 0
  store double %v227, ptr %v243, align 8
  %v244 = getelementptr inbounds %4, ptr %a0, i32 0, i32 5, i32 0, i32 1
  store double %v233, ptr %v244, align 8
  %v245 = getelementptr inbounds %4, ptr %a0, i32 0, i32 5, i32 0, i32 2
  store double %v239, ptr %v245, align 8
  br label %b2
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #1

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
