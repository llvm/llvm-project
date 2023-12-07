; RUN: llc -march=hexagon -O3 -verify-machineinstrs < %s
; REQUIRES: asserts
; Expect clean compilation.

target triple = "hexagon"

%s.0 = type { i16, i16, [4 x i16], i16, i16, [3 x i16], [3 x [4 x i16]], [3 x i16], [2 x [2 x i16]], i16, i16, i16, i16, [2 x i16], i16, i16, [3 x i16], [17 x i16] }

@g0 = external global i16
@g1 = external global [2 x i16]
@g2 = external global [10 x i16]
@g3 = external global %s.0
@g4 = external global [160 x i16]
@g5 = external global i16
@g6 = external global i16
@g7 = external global i16
@g8 = external global i16
@g9 = external global i16
@g10 = external global i16
@g11 = external global i16
@g12 = external global [192 x i16]
@g13 = external global [10 x i32]
@g14 = external global i16

; Function Attrs: nounwind
define signext i16 @f0(i16 signext %a0, ptr nocapture readonly %a1) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  store i32 327685, ptr %v0, align 4
  store i32 1048592, ptr %v1, align 4
  %v2 = sext i16 %a0 to i32
  switch i32 %v2, label %b35 [
    i32 0, label %b1
    i32 1, label %b9
    i32 2, label %b11
    i32 3, label %b15
    i32 4, label %b20
    i32 5, label %b30
  ]

b1:                                               ; preds = %b0
  %v3 = load i16, ptr %a1, align 2, !tbaa !0
  %v4 = icmp eq i16 %v3, -1
  br i1 %v4, label %b2, label %b4

b2:                                               ; preds = %b1
  %v5 = load i16, ptr @g0, align 2, !tbaa !0
  %v6 = add i16 %v5, 1
  store i16 %v6, ptr @g0, align 2, !tbaa !0
  %v7 = icmp sgt i16 %v6, 2
  br i1 %v7, label %b3, label %b5

b3:                                               ; preds = %b2
  store i16 3, ptr @g0, align 2, !tbaa !0
  br label %b35

b4:                                               ; preds = %b1
  store i16 0, ptr @g0, align 2, !tbaa !0
  br label %b5

b5:                                               ; preds = %b4, %b2
  %v8 = load i16, ptr %a1, align 2, !tbaa !0
  %v9 = icmp ne i16 %v8, 0
  %v10 = load i16, ptr @g1, align 2
  %v11 = icmp eq i16 %v10, 0
  %v12 = and i1 %v9, %v11
  br i1 %v12, label %b6, label %b35

b6:                                               ; preds = %b5
  call void @f1(ptr %v0, ptr %v1, ptr @g2, ptr getelementptr inbounds (%s.0, ptr @g3, i32 0, i32 2, i32 0), ptr @g4)
  %v15 = load i16, ptr @g2, align 2, !tbaa !0
  %v16 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 1), align 2, !tbaa !0
  %v17 = icmp sgt i16 %v15, %v16
  %v18 = select i1 %v17, i16 %v15, i16 %v16
  %v19 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 2), align 2, !tbaa !0
  %v20 = icmp sgt i16 %v18, %v19
  %v21 = select i1 %v20, i16 %v18, i16 %v19
  %v22 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 3), align 2, !tbaa !0
  %v23 = icmp sgt i16 %v21, %v22
  %v24 = select i1 %v23, i16 %v21, i16 %v22
  %v25 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 4), align 2, !tbaa !0
  %v26 = icmp sle i16 %v24, %v25
  %v27 = xor i1 %v23, true
  %v28 = or i1 %v26, %v27
  %v29 = select i1 %v26, i16 %v25, i16 %v22
  %v30 = select i1 %v28, i16 %v29, i16 %v21
  %v31 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 5), align 2, !tbaa !0
  %v32 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 6), align 2, !tbaa !0
  %v33 = icmp slt i16 %v31, %v32
  %v34 = select i1 %v33, i16 %v31, i16 %v32
  %v35 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 7), align 2, !tbaa !0
  %v36 = icmp slt i16 %v34, %v35
  %v37 = select i1 %v36, i16 %v34, i16 %v35
  %v38 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 8), align 2, !tbaa !0
  %v39 = icmp slt i16 %v37, %v38
  %v40 = select i1 %v39, i16 %v37, i16 %v38
  %v41 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 9), align 2, !tbaa !0
  %v42 = icmp sge i16 %v40, %v41
  %v43 = xor i1 %v39, true
  %v44 = or i1 %v42, %v43
  %v45 = select i1 %v42, i16 %v41, i16 %v38
  %v46 = select i1 %v44, i16 %v45, i16 %v37
  %v47 = icmp slt i16 %v30, %v46
  br i1 %v47, label %b7, label %b35

b7:                                               ; preds = %b6
  %v48 = load i16, ptr @g5, align 2, !tbaa !0
  %v49 = icmp eq i16 %v48, 4
  %v50 = load i16, ptr @g6, align 2
  %v51 = icmp eq i16 %v50, 0
  %v52 = and i1 %v49, %v51
  br i1 %v52, label %b35, label %b8

b8:                                               ; preds = %b7
  br label %b35

b9:                                               ; preds = %b0
  store i16 0, ptr @g0, align 2, !tbaa !0
  %v53 = load i16, ptr %a1, align 2, !tbaa !0
  %v54 = icmp eq i16 %v53, 0
  %v55 = zext i1 %v54 to i16
  %v56 = getelementptr i16, ptr %a1, i32 1
  %v57 = load i16, ptr %v56, align 2, !tbaa !0
  %v58 = icmp eq i16 %v57, 0
  %v59 = zext i1 %v58 to i16
  %v60 = add nuw nsw i16 %v59, %v55
  %v61 = getelementptr inbounds i16, ptr %a1, i32 2
  %v62 = load i16, ptr %v61, align 2, !tbaa !0
  %v63 = icmp ult i16 %v62, 256
  %v64 = zext i1 %v63 to i16
  %v65 = add nuw nsw i16 %v64, %v60
  %v66 = load i16, ptr @g1, align 2
  %v67 = icmp eq i16 %v65, 3
  %v68 = icmp ne i16 %v66, 0
  %v69 = or i1 %v68, %v67
  %v70 = load i16, ptr getelementptr inbounds (%s.0, ptr @g3, i32 0, i32 9), align 2
  %v71 = icmp eq i16 %v70, 3
  %v72 = or i1 %v71, %v69
  br i1 %v72, label %b35, label %b10

b10:                                              ; preds = %b9
  br label %b35

b11:                                              ; preds = %b0
  store i16 0, ptr @g0, align 2, !tbaa !0
  %v73 = load i16, ptr %a1, align 2, !tbaa !0
  %v74 = icmp eq i16 %v73, 0
  %v75 = zext i1 %v74 to i16
  %v76 = getelementptr i16, ptr %a1, i32 1
  %v77 = load i16, ptr %v76, align 2, !tbaa !0
  %v78 = icmp eq i16 %v77, 0
  %v79 = zext i1 %v78 to i16
  %v80 = add nuw nsw i16 %v79, %v75
  %v81 = getelementptr inbounds i16, ptr %a1, i32 2
  %v82 = load i16, ptr %v81, align 2, !tbaa !0
  %v83 = icmp ult i16 %v82, 256
  %v84 = zext i1 %v83 to i16
  %v85 = add nuw nsw i16 %v84, %v80
  %v86 = icmp ne i16 %v85, 3
  %v87 = load i16, ptr @g1, align 2
  %v88 = icmp eq i16 %v87, 0
  %v89 = and i1 %v88, %v86
  br i1 %v89, label %b12, label %b35

b12:                                              ; preds = %b11
  %v90 = load i16, ptr @g5, align 2, !tbaa !0
  switch i16 %v90, label %b14 [
    i16 1, label %b35
    i16 2, label %b13
  ]

b13:                                              ; preds = %b12
  %v91 = load i16, ptr @g7, align 2, !tbaa !0
  %v92 = load i16, ptr @g6, align 2
  %v93 = or i16 %v92, %v91
  %v94 = icmp eq i16 %v93, 0
  br i1 %v94, label %b35, label %b14

b14:                                              ; preds = %b13, %b12
  br label %b35

b15:                                              ; preds = %b0
  store i16 0, ptr @g0, align 2, !tbaa !0
  %v95 = load i16, ptr %a1, align 2, !tbaa !0
  %v96 = icmp eq i16 %v95, 0
  %v97 = zext i1 %v96 to i16
  %v98 = getelementptr i16, ptr %a1, i32 1
  %v99 = load i16, ptr %v98, align 2, !tbaa !0
  %v100 = icmp eq i16 %v99, 0
  %v101 = zext i1 %v100 to i16
  %v102 = add nuw nsw i16 %v101, %v97
  %v103 = getelementptr i16, ptr %a1, i32 2
  %v104 = load i16, ptr %v103, align 2, !tbaa !0
  %v105 = icmp eq i16 %v104, 0
  %v106 = zext i1 %v105 to i16
  %v107 = add nuw nsw i16 %v106, %v102
  %v108 = getelementptr i16, ptr %a1, i32 3
  %v109 = load i16, ptr %v108, align 2, !tbaa !0
  %v110 = icmp eq i16 %v109, 0
  %v111 = zext i1 %v110 to i16
  %v112 = add nuw nsw i16 %v111, %v107
  %v113 = getelementptr i16, ptr %a1, i32 4
  %v114 = load i16, ptr %v113, align 2, !tbaa !0
  %v115 = icmp eq i16 %v114, 0
  %v116 = zext i1 %v115 to i16
  %v117 = add nuw nsw i16 %v116, %v112
  %v118 = icmp eq i16 %v117, 5
  br i1 %v118, label %b35, label %b16

b16:                                              ; preds = %b15
  %v119 = load i16, ptr getelementptr inbounds (%s.0, ptr @g3, i32 0, i32 3), align 2, !tbaa !4
  switch i16 %v119, label %b17 [
    i16 120, label %b19
    i16 115, label %b19
  ]

b17:                                              ; preds = %b16
  %v120 = icmp sgt i16 %v119, 100
  br i1 %v120, label %b35, label %b18

b18:                                              ; preds = %b17
  tail call void @f2(ptr getelementptr inbounds (%s.0, ptr @g3, i32 0, i32 2, i32 0), ptr @g2)
  %v121 = load i16, ptr @g2, align 2, !tbaa !0
  %v122 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 1), align 2, !tbaa !0
  %v123 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 2), align 2, !tbaa !0
  %v124 = icmp sgt i16 %v122, %v123
  %v125 = select i1 %v124, i16 %v122, i16 %v123
  %v126 = icmp sgt i16 %v121, %v125
  %v127 = select i1 %v126, i16 %v121, i16 %v125
  %v128 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 6), align 2, !tbaa !0
  %v129 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 7), align 2, !tbaa !0
  %v130 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 8), align 2, !tbaa !0
  %v131 = icmp slt i16 %v129, %v130
  %v132 = select i1 %v131, i16 %v129, i16 %v130
  %v133 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 9), align 2, !tbaa !0
  %v134 = icmp slt i16 %v132, %v133
  %v135 = select i1 %v134, i16 %v132, i16 %v133
  %v136 = icmp slt i16 %v128, %v135
  %v137 = select i1 %v136, i16 %v128, i16 %v135
  %v138 = icmp slt i16 %v127, %v137
  br i1 %v138, label %b19, label %b35

b19:                                              ; preds = %b18, %b16, %b16
  br label %b35

b20:                                              ; preds = %b0
  store i16 0, ptr @g0, align 2, !tbaa !0
  %v139 = load i16, ptr %a1, align 2, !tbaa !0
  %v140 = icmp eq i16 %v139, 0
  %v141 = zext i1 %v140 to i16
  %v142 = getelementptr i16, ptr %a1, i32 1
  %v143 = load i16, ptr %v142, align 2, !tbaa !0
  %v144 = icmp eq i16 %v143, 0
  %v145 = zext i1 %v144 to i16
  %v146 = add nuw nsw i16 %v145, %v141
  %v147 = getelementptr i16, ptr %a1, i32 2
  %v148 = load i16, ptr %v147, align 2, !tbaa !0
  %v149 = icmp eq i16 %v148, 0
  %v150 = zext i1 %v149 to i16
  %v151 = add nuw nsw i16 %v150, %v146
  %v152 = getelementptr i16, ptr %a1, i32 3
  %v153 = load i16, ptr %v152, align 2, !tbaa !0
  %v154 = icmp eq i16 %v153, 0
  %v155 = zext i1 %v154 to i16
  %v156 = add nuw nsw i16 %v155, %v151
  %v157 = getelementptr i16, ptr %a1, i32 4
  %v158 = load i16, ptr %v157, align 2, !tbaa !0
  %v159 = icmp eq i16 %v158, 0
  %v160 = zext i1 %v159 to i16
  %v161 = add nuw nsw i16 %v160, %v156
  %v162 = getelementptr i16, ptr %a1, i32 5
  %v163 = load i16, ptr %v162, align 2, !tbaa !0
  %v164 = icmp eq i16 %v163, 0
  %v165 = zext i1 %v164 to i16
  %v166 = add nuw nsw i16 %v165, %v161
  %v167 = getelementptr i16, ptr %a1, i32 6
  %v168 = load i16, ptr %v167, align 2, !tbaa !0
  %v169 = icmp eq i16 %v168, 0
  %v170 = zext i1 %v169 to i16
  %v171 = add nuw nsw i16 %v170, %v166
  %v172 = getelementptr i16, ptr %a1, i32 7
  %v173 = load i16, ptr %v172, align 2, !tbaa !0
  %v174 = icmp eq i16 %v173, 0
  %v175 = zext i1 %v174 to i16
  %v176 = add i16 %v175, %v171
  %v177 = getelementptr i16, ptr %a1, i32 8
  %v178 = load i16, ptr %v177, align 2, !tbaa !0
  %v179 = icmp eq i16 %v178, 0
  %v180 = zext i1 %v179 to i16
  %v181 = add i16 %v180, %v176
  %v182 = getelementptr i16, ptr %a1, i32 9
  %v183 = load i16, ptr %v182, align 2, !tbaa !0
  %v184 = icmp eq i16 %v183, 0
  %v185 = zext i1 %v184 to i16
  %v186 = add i16 %v185, %v181
  %v187 = getelementptr inbounds i16, ptr %a1, i32 10
  %v188 = load i16, ptr %v187, align 2, !tbaa !0
  %v189 = icmp ult i16 %v188, 32
  %v190 = zext i1 %v189 to i16
  %v191 = add i16 %v190, %v186
  %v192 = icmp eq i16 %v191, 11
  br i1 %v192, label %b35, label %b21

b21:                                              ; preds = %b20
  tail call void @f3(ptr getelementptr inbounds (%s.0, ptr @g3, i32 0, i32 2, i32 0), ptr @g2)
  %v193 = load i16, ptr @g8, align 2, !tbaa !0
  %v194 = icmp eq i16 %v193, 0
  br i1 %v194, label %b22, label %b35

b22:                                              ; preds = %b21
  %v195 = load i16, ptr getelementptr inbounds (%s.0, ptr @g3, i32 0, i32 3), align 2, !tbaa !4
  %v196 = icmp sgt i16 %v195, 100
  br i1 %v196, label %b35, label %b23

b23:                                              ; preds = %b22
  %v197 = load i16, ptr @g2, align 2, !tbaa !0
  %v198 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 1), align 2, !tbaa !0
  %v199 = icmp sgt i16 %v197, %v198
  %v200 = select i1 %v199, i16 %v197, i16 %v198
  %v201 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 4), align 2, !tbaa !0
  %v202 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 5), align 2, !tbaa !0
  %v203 = icmp slt i16 %v201, %v202
  %v204 = select i1 %v203, i16 %v201, i16 %v202
  %v205 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 6), align 2, !tbaa !0
  %v206 = icmp slt i16 %v204, %v205
  %v207 = select i1 %v206, i16 %v204, i16 %v205
  %v208 = icmp slt i16 %v200, %v207
  br i1 %v208, label %b24, label %b35

b24:                                              ; preds = %b23
  %v209 = load i16, ptr @g5, align 2, !tbaa !0
  switch i16 %v209, label %b26 [
    i16 1, label %b35
    i16 2, label %b25
  ]

b25:                                              ; preds = %b24
  %v210 = load i16, ptr @g7, align 2, !tbaa !0
  %v211 = load i16, ptr @g6, align 2
  %v212 = or i16 %v211, %v210
  %v213 = icmp eq i16 %v212, 0
  br i1 %v213, label %b35, label %b27

b26:                                              ; preds = %b24
  %v214 = load i16, ptr @g6, align 2
  %v215 = icmp eq i16 %v214, 0
  br i1 %v215, label %b28, label %b35

b27:                                              ; preds = %b25
  %v216 = load i16, ptr @g9, align 2
  %v217 = icmp eq i16 %v216, 0
  br i1 %v217, label %b28, label %b35

b28:                                              ; preds = %b27, %b26
  %v218 = tail call signext i16 @f4(i16 signext %v195, i16 signext 20)
  store i16 %v218, ptr @g10, align 2, !tbaa !0
  %v219 = load i16, ptr @g11, align 2, !tbaa !0
  %v220 = tail call signext i16 @f6(i16 signext %v218, i16 signext %v219)
  %v221 = tail call signext i16 @f5(i16 signext %v220)
  %v222 = icmp sgt i16 %v221, 15
  br i1 %v222, label %b29, label %b35

b29:                                              ; preds = %b28
  call void @llvm.memset.p0.i32(ptr align 2 @g12, i8 0, i32 256, i1 false)
  call void @llvm.memset.p0.i32(ptr align 4 @g13, i8 0, i32 40, i1 false)
  tail call void @f7()
  br label %b35

b30:                                              ; preds = %b0
  store i16 0, ptr @g0, align 2, !tbaa !0
  %v223 = load i16, ptr %a1, align 2, !tbaa !0
  %v224 = icmp eq i16 %v223, 0
  %v225 = zext i1 %v224 to i16
  %v226 = getelementptr i16, ptr %a1, i32 1
  %v227 = load i16, ptr %v226, align 2, !tbaa !0
  %v228 = icmp eq i16 %v227, 0
  %v229 = zext i1 %v228 to i16
  %v230 = add nuw nsw i16 %v229, %v225
  %v231 = getelementptr i16, ptr %a1, i32 2
  %v232 = load i16, ptr %v231, align 2, !tbaa !0
  %v233 = icmp eq i16 %v232, 0
  %v234 = zext i1 %v233 to i16
  %v235 = add nuw nsw i16 %v234, %v230
  %v236 = getelementptr i16, ptr %a1, i32 3
  %v237 = load i16, ptr %v236, align 2, !tbaa !0
  %v238 = icmp eq i16 %v237, 0
  %v239 = zext i1 %v238 to i16
  %v240 = add nuw nsw i16 %v239, %v235
  %v241 = getelementptr i16, ptr %a1, i32 4
  %v242 = load i16, ptr %v241, align 2, !tbaa !0
  %v243 = icmp eq i16 %v242, 0
  %v244 = zext i1 %v243 to i16
  %v245 = add nuw nsw i16 %v244, %v240
  %v246 = getelementptr i16, ptr %a1, i32 5
  %v247 = load i16, ptr %v246, align 2, !tbaa !0
  %v248 = icmp eq i16 %v247, 0
  %v249 = zext i1 %v248 to i16
  %v250 = add nuw nsw i16 %v249, %v245
  %v251 = getelementptr i16, ptr %a1, i32 6
  %v252 = load i16, ptr %v251, align 2, !tbaa !0
  %v253 = icmp eq i16 %v252, 0
  %v254 = zext i1 %v253 to i16
  %v255 = add nuw nsw i16 %v254, %v250
  %v256 = getelementptr i16, ptr %a1, i32 7
  %v257 = load i16, ptr %v256, align 2, !tbaa !0
  %v258 = icmp eq i16 %v257, 0
  %v259 = zext i1 %v258 to i16
  %v260 = add i16 %v259, %v255
  %v261 = getelementptr i16, ptr %a1, i32 8
  %v262 = load i16, ptr %v261, align 2, !tbaa !0
  %v263 = icmp eq i16 %v262, 0
  %v264 = zext i1 %v263 to i16
  %v265 = add i16 %v264, %v260
  %v266 = getelementptr i16, ptr %a1, i32 9
  %v267 = load i16, ptr %v266, align 2, !tbaa !0
  %v268 = icmp eq i16 %v267, 0
  %v269 = zext i1 %v268 to i16
  %v270 = add i16 %v269, %v265
  %v271 = getelementptr inbounds i16, ptr %a1, i32 10
  %v272 = load i16, ptr %v271, align 2, !tbaa !0
  %v273 = icmp ult i16 %v272, 32
  %v274 = zext i1 %v273 to i16
  %v275 = add i16 %v274, %v270
  %v276 = icmp eq i16 %v275, 11
  br i1 %v276, label %b35, label %b31

b31:                                              ; preds = %b30
  tail call void @f3(ptr getelementptr inbounds (%s.0, ptr @g3, i32 0, i32 2, i32 0), ptr @g2)
  %v277 = load i16, ptr @g14, align 2, !tbaa !0
  %v278 = icmp eq i16 %v277, 0
  br i1 %v278, label %b32, label %b34

b32:                                              ; preds = %b31
  %v279 = load i16, ptr getelementptr inbounds (%s.0, ptr @g3, i32 0, i32 3), align 2, !tbaa !4
  %v280 = icmp sgt i16 %v279, 100
  br i1 %v280, label %b35, label %b33

b33:                                              ; preds = %b32
  %v281 = load i16, ptr @g2, align 2, !tbaa !0
  %v282 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 1), align 2, !tbaa !0
  %v283 = icmp sgt i16 %v281, %v282
  %v284 = select i1 %v283, i16 %v281, i16 %v282
  %v285 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 4), align 2, !tbaa !0
  %v286 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 5), align 2, !tbaa !0
  %v287 = icmp slt i16 %v285, %v286
  %v288 = select i1 %v287, i16 %v285, i16 %v286
  %v289 = load i16, ptr getelementptr inbounds ([10 x i16], ptr @g2, i32 0, i32 6), align 2, !tbaa !0
  %v290 = icmp slt i16 %v288, %v289
  %v291 = select i1 %v290, i16 %v288, i16 %v289
  %v292 = icmp slt i16 %v284, %v291
  br i1 %v292, label %b34, label %b35

b34:                                              ; preds = %b33, %b31
  br label %b35

b35:                                              ; preds = %b34, %b33, %b32, %b30, %b29, %b28, %b27, %b26, %b25, %b24, %b23, %b22, %b21, %b20, %b19, %b18, %b17, %b15, %b14, %b13, %b12, %b11, %b10, %b9, %b8, %b7, %b6, %b5, %b3, %b0
  %v293 = phi i16 [ 0, %b34 ], [ 1, %b29 ], [ 0, %b19 ], [ 0, %b14 ], [ 0, %b10 ], [ 0, %b3 ], [ 0, %b8 ], [ 1, %b5 ], [ 1, %b6 ], [ 1, %b9 ], [ 1, %b11 ], [ 1, %b12 ], [ 1, %b15 ], [ 1, %b17 ], [ 1, %b18 ], [ 1, %b20 ], [ 1, %b22 ], [ 1, %b23 ], [ 1, %b24 ], [ 0, %b27 ], [ 0, %b28 ], [ 0, %b21 ], [ 1, %b30 ], [ 1, %b32 ], [ 1, %b33 ], [ 0, %b0 ], [ 1, %b7 ], [ 1, %b13 ], [ 1, %b25 ], [ 0, %b26 ]
  ret i16 %v293
}

; Function Attrs: nounwind
declare void @f1(ptr, ptr, ptr, ptr, ptr) #0

; Function Attrs: nounwind
declare void @f2(ptr, ptr) #0

; Function Attrs: nounwind
declare void @f3(ptr, ptr) #0

; Function Attrs: nounwind
declare signext i16 @f4(i16 signext, i16 signext) #0

; Function Attrs: nounwind
declare signext i16 @f5(i16 signext) #0

; Function Attrs: nounwind
declare signext i16 @f6(i16 signext, i16 signext) #0

; Function Attrs: nounwind
declare void @f7() #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !1, i64 12}
!5 = !{!"_ZTS6PACKET", !1, i64 0, !1, i64 2, !2, i64 4, !1, i64 12, !1, i64 14, !2, i64 16, !2, i64 22, !2, i64 46, !2, i64 52, !1, i64 60, !1, i64 62, !1, i64 64, !1, i64 66, !2, i64 68, !1, i64 72, !1, i64 74, !2, i64 76, !2, i64 82}
