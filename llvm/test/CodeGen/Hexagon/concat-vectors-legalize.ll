; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK-NOT: setbit(r{{[0-9]+}},#1)

target triple = "hexagon-unknown--elf"

%s.8 = type { ptr, i32, i32, i32, i32, ptr, ptr, ptr }
%s.9 = type { %s.10 }
%s.10 = type { i64 }
%s.4 = type { i64, ptr, [4 x i32], [4 x i32], [4 x i32], i32, i8, i8, [6 x i8] }

@g0 = private constant [6 x i8] c"input\00", align 32
@g1 = private constant [11 x i8] c"gaussian11\00", align 32
@g2 = private constant [2 x %s.8] [%s.8 { ptr @g0, i32 1, i32 2, i32 1, i32 8, ptr null, ptr null, ptr null }, %s.8 { ptr @g1, i32 2, i32 2, i32 1, i32 8, ptr null, ptr null, ptr null }]
@g3 = private constant [53 x i8] c"hexagon-32-os_unknown-no_asserts-no_bounds_query-hvx\00", align 32

; Function Attrs: nounwind
declare ptr @f0(ptr, i32) #0

; Function Attrs: nounwind
declare void @f1(ptr, ptr) #0

; Function Attrs: nounwind
declare noalias ptr @f2(ptr, i32) #0

; Function Attrs: nounwind
declare void @f3(ptr, ptr) #0

; Function Attrs: nounwind
declare void @f4() #0

; Function Attrs: nounwind
declare void @f5() #0

; Function Attrs: nounwind
define i32 @f6(ptr noalias nocapture readonly %a0, ptr noalias nocapture readonly %a1) #0 {
b0:
  %v0 = getelementptr inbounds %s.4, ptr %a0, i32 0, i32 1
  %v1 = load ptr, ptr %v0
  %v2 = getelementptr inbounds %s.4, ptr %a0, i32 0, i32 2, i32 0
  %v3 = load i32, ptr %v2
  %v4 = getelementptr inbounds %s.4, ptr %a0, i32 0, i32 2, i32 1
  %v5 = load i32, ptr %v4
  %v6 = getelementptr inbounds %s.4, ptr %a0, i32 0, i32 3, i32 1
  %v7 = load i32, ptr %v6
  %v8 = getelementptr inbounds %s.4, ptr %a0, i32 0, i32 4, i32 0
  %v9 = load i32, ptr %v8
  %v10 = getelementptr inbounds %s.4, ptr %a0, i32 0, i32 4, i32 1
  %v11 = load i32, ptr %v10
  %v12 = getelementptr inbounds %s.4, ptr %a1, i32 0, i32 1
  %v13 = load ptr, ptr %v12
  %v14 = getelementptr inbounds %s.4, ptr %a1, i32 0, i32 2, i32 0
  %v15 = load i32, ptr %v14
  %v16 = getelementptr inbounds %s.4, ptr %a1, i32 0, i32 2, i32 1
  %v17 = load i32, ptr %v16
  %v18 = getelementptr inbounds %s.4, ptr %a1, i32 0, i32 3, i32 1
  %v19 = load i32, ptr %v18
  %v20 = getelementptr inbounds %s.4, ptr %a1, i32 0, i32 4, i32 0
  %v21 = load i32, ptr %v20
  %v22 = getelementptr inbounds %s.4, ptr %a1, i32 0, i32 4, i32 1
  %v23 = load i32, ptr %v22
  %v24 = add nsw i32 %v21, %v15
  %v25 = add nsw i32 %v24, -64
  %v26 = icmp slt i32 %v21, %v25
  %v27 = select i1 %v26, i32 %v21, i32 %v25
  %v28 = add nsw i32 %v15, -1
  %v29 = and i32 %v28, -64
  %v30 = add i32 %v21, 63
  %v31 = add i32 %v30, %v29
  %v32 = add nsw i32 %v24, -1
  %v33 = icmp slt i32 %v31, %v32
  %v34 = select i1 %v33, i32 %v31, i32 %v32
  %v35 = sub nsw i32 %v34, %v27
  %v36 = icmp slt i32 %v24, %v34
  %v37 = select i1 %v36, i32 %v34, i32 %v24
  %v38 = add nsw i32 %v37, -1
  %v39 = icmp slt i32 %v38, %v34
  %v40 = select i1 %v39, i32 %v34, i32 %v38
  %v41 = add nsw i32 %v17, 1
  %v42 = sext i32 %v41 to i64
  %v43 = sub nsw i32 %v40, %v27
  %v44 = add nsw i32 %v43, 2
  %v45 = sext i32 %v44 to i64
  %v46 = mul nsw i64 %v45, %v42
  %v47 = trunc i64 %v46 to i32
  %v48 = tail call ptr @f2(ptr null, i32 %v47)
  %v49 = add nsw i32 %v23, -1
  %v50 = add i32 %v23, %v17
  %v51 = icmp sgt i32 %v23, %v50
  br i1 %v51, label %b12, label %b1, !prof !3

b1:                                               ; preds = %b11, %b0
  %v52 = phi i32 [ %v220, %b11 ], [ %v49, %b0 ]
  %v53 = icmp slt i32 %v9, %v24
  %v54 = select i1 %v53, i32 %v9, i32 %v24
  %v55 = add nsw i32 %v21, -1
  %v56 = icmp slt i32 %v54, %v55
  %v57 = select i1 %v56, i32 %v55, i32 %v54
  %v58 = add nsw i32 %v9, %v3
  %v59 = icmp slt i32 %v58, %v24
  %v60 = select i1 %v59, i32 %v58, i32 %v24
  %v61 = icmp slt i32 %v60, %v57
  %v62 = select i1 %v61, i32 %v57, i32 %v60
  %v63 = icmp slt i32 %v57, %v21
  br i1 %v63, label %b7, label %b2, !prof !3

b2:                                               ; preds = %b1
  %v64 = add nsw i32 %v11, %v5
  %v65 = add nsw i32 %v64, -1
  %v66 = icmp slt i32 %v52, %v65
  br i1 %v66, label %b3, label %b4

b3:                                               ; preds = %b3, %b2
  %v67 = phi i32 [ %v96, %b3 ], [ %v55, %b2 ]
  %v68 = mul nsw i32 %v11, %v7
  %v69 = icmp slt i32 %v52, %v11
  %v70 = select i1 %v69, i32 %v11, i32 %v52
  %v71 = mul nsw i32 %v70, %v7
  %v72 = add nsw i32 %v58, -1
  %v73 = icmp slt i32 %v67, %v72
  %v74 = select i1 %v73, i32 %v67, i32 %v72
  %v75 = icmp slt i32 %v74, %v9
  %v76 = select i1 %v75, i32 %v9, i32 %v74
  %v77 = add i32 %v68, %v9
  %v78 = sub i32 %v71, %v77
  %v79 = add i32 %v78, %v76
  %v80 = getelementptr inbounds i8, ptr %v1, i32 %v79
  %v81 = load i8, ptr %v80, align 1, !tbaa !4
  %v82 = icmp sle i32 %v64, %v52
  %v83 = icmp sle i32 %v58, %v67
  %v84 = icmp slt i32 %v67, %v9
  %v85 = or i1 %v84, %v83
  %v86 = or i1 %v69, %v85
  %v87 = or i1 %v82, %v86
  %v88 = select i1 %v87, i8 0, i8 %v81
  %v89 = sub i32 1, %v23
  %v90 = add i32 %v89, %v52
  %v91 = mul nsw i32 %v90, %v44
  %v92 = sub i32 1, %v27
  %v93 = add i32 %v92, %v91
  %v94 = add i32 %v93, %v67
  %v95 = getelementptr inbounds i8, ptr %v48, i32 %v94
  store i8 %v88, ptr %v95, align 1, !tbaa !7
  %v96 = add nsw i32 %v67, 1
  %v97 = icmp eq i32 %v96, %v57
  br i1 %v97, label %b7, label %b3

b4:                                               ; preds = %b2
  %v98 = icmp slt i32 %v5, 1
  br i1 %v98, label %b5, label %b6

b5:                                               ; preds = %b5, %b4
  %v99 = phi i32 [ %v123, %b5 ], [ %v55, %b4 ]
  %v100 = add nsw i32 %v58, -1
  %v101 = icmp slt i32 %v99, %v100
  %v102 = select i1 %v101, i32 %v99, i32 %v100
  %v103 = icmp slt i32 %v102, %v9
  %v104 = select i1 %v103, i32 %v9, i32 %v102
  %v105 = sub i32 %v104, %v9
  %v106 = getelementptr inbounds i8, ptr %v1, i32 %v105
  %v107 = load i8, ptr %v106, align 1, !tbaa !4
  %v108 = icmp sle i32 %v64, %v52
  %v109 = icmp slt i32 %v52, %v11
  %v110 = icmp sle i32 %v58, %v99
  %v111 = icmp slt i32 %v99, %v9
  %v112 = or i1 %v111, %v110
  %v113 = or i1 %v109, %v112
  %v114 = or i1 %v108, %v113
  %v115 = select i1 %v114, i8 0, i8 %v107
  %v116 = sub i32 1, %v23
  %v117 = add i32 %v116, %v52
  %v118 = mul nsw i32 %v117, %v44
  %v119 = sub i32 1, %v27
  %v120 = add i32 %v119, %v118
  %v121 = add i32 %v120, %v99
  %v122 = getelementptr inbounds i8, ptr %v48, i32 %v121
  store i8 %v115, ptr %v122, align 1, !tbaa !7
  %v123 = add nsw i32 %v99, 1
  %v124 = icmp eq i32 %v123, %v57
  br i1 %v124, label %b7, label %b5

b6:                                               ; preds = %b6, %b4
  %v125 = phi i32 [ %v153, %b6 ], [ %v55, %b4 ]
  %v126 = mul nsw i32 %v11, %v7
  %v127 = mul nsw i32 %v65, %v7
  %v128 = add nsw i32 %v58, -1
  %v129 = icmp slt i32 %v125, %v128
  %v130 = select i1 %v129, i32 %v125, i32 %v128
  %v131 = icmp slt i32 %v130, %v9
  %v132 = select i1 %v131, i32 %v9, i32 %v130
  %v133 = add i32 %v126, %v9
  %v134 = sub i32 %v127, %v133
  %v135 = add i32 %v134, %v132
  %v136 = getelementptr inbounds i8, ptr %v1, i32 %v135
  %v137 = load i8, ptr %v136, align 1, !tbaa !4
  %v138 = icmp sle i32 %v64, %v52
  %v139 = icmp slt i32 %v52, %v11
  %v140 = icmp sle i32 %v58, %v125
  %v141 = icmp slt i32 %v125, %v9
  %v142 = or i1 %v141, %v140
  %v143 = or i1 %v139, %v142
  %v144 = or i1 %v138, %v143
  %v145 = select i1 %v144, i8 0, i8 %v137
  %v146 = sub i32 1, %v23
  %v147 = add i32 %v146, %v52
  %v148 = mul nsw i32 %v147, %v44
  %v149 = sub i32 1, %v27
  %v150 = add i32 %v149, %v148
  %v151 = add i32 %v150, %v125
  %v152 = getelementptr inbounds i8, ptr %v48, i32 %v151
  store i8 %v145, ptr %v152, align 1, !tbaa !7
  %v153 = add nsw i32 %v125, 1
  %v154 = icmp eq i32 %v153, %v57
  br i1 %v154, label %b7, label %b6

b7:                                               ; preds = %b6, %b5, %b3, %b1
  %v155 = icmp slt i32 %v57, %v62
  br i1 %v155, label %b8, label %b9, !prof !9

b8:                                               ; preds = %b8, %b7
  %v156 = phi i32 [ %v181, %b8 ], [ %v57, %b7 ]
  %v157 = mul nsw i32 %v11, %v7
  %v158 = add nsw i32 %v11, %v5
  %v159 = add nsw i32 %v158, -1
  %v160 = icmp slt i32 %v52, %v159
  %v161 = select i1 %v160, i32 %v52, i32 %v159
  %v162 = icmp slt i32 %v161, %v11
  %v163 = select i1 %v162, i32 %v11, i32 %v161
  %v164 = mul nsw i32 %v163, %v7
  %v165 = add i32 %v157, %v9
  %v166 = sub i32 %v164, %v165
  %v167 = add i32 %v166, %v156
  %v168 = getelementptr inbounds i8, ptr %v1, i32 %v167
  %v169 = load i8, ptr %v168, align 1, !tbaa !4
  %v170 = icmp sle i32 %v158, %v52
  %v171 = icmp slt i32 %v52, %v11
  %v172 = or i1 %v171, %v170
  %v173 = select i1 %v172, i8 0, i8 %v169
  %v174 = sub i32 1, %v23
  %v175 = add i32 %v174, %v52
  %v176 = mul nsw i32 %v175, %v44
  %v177 = sub i32 1, %v27
  %v178 = add i32 %v177, %v176
  %v179 = add i32 %v178, %v156
  %v180 = getelementptr inbounds i8, ptr %v48, i32 %v179
  store i8 %v173, ptr %v180, align 1, !tbaa !7
  %v181 = add nsw i32 %v156, 1
  %v182 = icmp eq i32 %v181, %v62
  br i1 %v182, label %b9, label %b8

b9:                                               ; preds = %b8, %b7
  %v183 = icmp slt i32 %v62, %v24
  br i1 %v183, label %b10, label %b11, !prof !9

b10:                                              ; preds = %b10, %b9
  %v184 = phi i32 [ %v218, %b10 ], [ %v62, %b9 ]
  %v185 = mul nsw i32 %v11, %v7
  %v186 = add nsw i32 %v11, %v5
  %v187 = add nsw i32 %v186, -1
  %v188 = icmp slt i32 %v52, %v187
  %v189 = select i1 %v188, i32 %v52, i32 %v187
  %v190 = icmp slt i32 %v189, %v11
  %v191 = select i1 %v190, i32 %v11, i32 %v189
  %v192 = mul nsw i32 %v191, %v7
  %v193 = add nsw i32 %v58, -1
  %v194 = icmp slt i32 %v184, %v193
  %v195 = select i1 %v194, i32 %v184, i32 %v193
  %v196 = icmp slt i32 %v195, %v9
  %v197 = select i1 %v196, i32 %v9, i32 %v195
  %v198 = add i32 %v185, %v9
  %v199 = sub i32 %v192, %v198
  %v200 = add i32 %v199, %v197
  %v201 = getelementptr inbounds i8, ptr %v1, i32 %v200
  %v202 = load i8, ptr %v201, align 1, !tbaa !4
  %v203 = icmp sle i32 %v186, %v52
  %v204 = icmp slt i32 %v52, %v11
  %v205 = icmp sle i32 %v58, %v184
  %v206 = icmp slt i32 %v184, %v9
  %v207 = or i1 %v206, %v205
  %v208 = or i1 %v204, %v207
  %v209 = or i1 %v203, %v208
  %v210 = select i1 %v209, i8 0, i8 %v202
  %v211 = sub i32 1, %v23
  %v212 = add i32 %v211, %v52
  %v213 = mul nsw i32 %v212, %v44
  %v214 = sub i32 1, %v27
  %v215 = add i32 %v214, %v213
  %v216 = add i32 %v215, %v184
  %v217 = getelementptr inbounds i8, ptr %v48, i32 %v216
  store i8 %v210, ptr %v217, align 1, !tbaa !7
  %v218 = add nsw i32 %v184, 1
  %v219 = icmp eq i32 %v218, %v24
  br i1 %v219, label %b11, label %b10

b11:                                              ; preds = %b10, %b9
  %v220 = add nsw i32 %v52, 1
  %v221 = icmp eq i32 %v220, %v50
  br i1 %v221, label %b12, label %b1

b12:                                              ; preds = %b11, %b0
  %v222 = add nsw i32 %v35, 1
  %v223 = sext i32 %v222 to i64
  %v224 = shl nsw i64 %v42, 2
  %v225 = mul i64 %v224, %v223
  %v226 = trunc i64 %v225 to i32
  %v227 = tail call ptr @f2(ptr null, i32 %v226)
  br i1 %v51, label %b14, label %b13, !prof !3

b13:                                              ; preds = %b19, %b12
  %v228 = phi i32 [ %v351, %b19 ], [ %v49, %b12 ]
  %v229 = ashr i32 %v15, 6
  %v230 = icmp slt i32 %v229, 0
  %v231 = select i1 %v230, i32 0, i32 %v229
  %v232 = icmp sgt i32 %v231, 0
  br i1 %v232, label %b16, label %b17, !prof !9

b14:                                              ; preds = %b19, %b12
  %v233 = icmp eq ptr %v48, null
  br i1 %v233, label %b20, label %b15

b15:                                              ; preds = %b14
  tail call void @f3(ptr null, ptr %v48) #2
  br label %b20

b16:                                              ; preds = %b16, %b13
  %v234 = phi i32 [ %v289, %b16 ], [ 0, %b13 ]
  %v235 = sub nsw i32 %v228, %v23
  %v236 = add nsw i32 %v235, 1
  %v237 = mul nsw i32 %v236, %v44
  %v238 = shl i32 %v234, 6
  %v239 = sub i32 %v21, %v27
  %v240 = add i32 %v239, %v238
  %v241 = add nsw i32 %v240, %v237
  %v242 = getelementptr inbounds i8, ptr %v48, i32 %v241
  %v244 = load <16 x i32>, ptr %v242, align 1, !tbaa !7
  %v245 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %v244)
  %v246 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v245)
  %v247 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v245)
  %v248 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %v247)
  %v249 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %v246)
  %v250 = add nsw i32 %v241, 1
  %v251 = getelementptr inbounds i8, ptr %v48, i32 %v250
  %v253 = load <16 x i32>, ptr %v251, align 1, !tbaa !7
  %v254 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %v253)
  %v255 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v254)
  %v256 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %v255)
  %v257 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v256)
  %v258 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v256)
  %v259 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v257, i32 168430090)
  %v260 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v258, i32 168430090)
  %v261 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v259, <16 x i32> %v260)
  %v262 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v254)
  %v263 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %v262)
  %v264 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v263)
  %v265 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v263)
  %v266 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v264, i32 168430090)
  %v267 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v265, i32 168430090)
  %v268 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v266, <16 x i32> %v267)
  %v269 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v248, <32 x i32> %v261)
  %v270 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v249, <32 x i32> %v268)
  %v271 = shufflevector <32 x i32> %v269, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v272 = mul nsw i32 %v236, %v222
  %v273 = add nsw i32 %v240, %v272
  %v275 = getelementptr inbounds i32, ptr %v227, i32 %v273
  store <16 x i32> %v271, ptr %v275, align 4, !tbaa !10
  %v277 = shufflevector <32 x i32> %v269, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v278 = add nsw i32 %v273, 16
  %v279 = getelementptr inbounds i32, ptr %v227, i32 %v278
  store <16 x i32> %v277, ptr %v279, align 4, !tbaa !10
  %v281 = shufflevector <32 x i32> %v270, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v282 = add nsw i32 %v273, 32
  %v283 = getelementptr inbounds i32, ptr %v227, i32 %v282
  store <16 x i32> %v281, ptr %v283, align 4, !tbaa !10
  %v285 = shufflevector <32 x i32> %v270, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v286 = add nsw i32 %v273, 48
  %v287 = getelementptr inbounds i32, ptr %v227, i32 %v286
  store <16 x i32> %v285, ptr %v287, align 4, !tbaa !10
  %v289 = add nuw nsw i32 %v234, 1
  %v290 = icmp eq i32 %v289, %v231
  br i1 %v290, label %b17, label %b16

b17:                                              ; preds = %b16, %b13
  %v291 = add nsw i32 %v15, 63
  %v292 = ashr i32 %v291, 6
  %v293 = icmp slt i32 %v231, %v292
  br i1 %v293, label %b18, label %b19, !prof !9

b18:                                              ; preds = %b18, %b17
  %v294 = phi i32 [ %v349, %b18 ], [ %v231, %b17 ]
  %v295 = sub nsw i32 %v228, %v23
  %v296 = add nsw i32 %v295, 1
  %v297 = mul nsw i32 %v296, %v44
  %v298 = sub nsw i32 %v24, %v27
  %v299 = add nsw i32 %v297, %v298
  %v300 = add nsw i32 %v299, -64
  %v301 = getelementptr inbounds i8, ptr %v48, i32 %v300
  %v303 = load <16 x i32>, ptr %v301, align 1, !tbaa !7
  %v304 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %v303)
  %v305 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v304)
  %v306 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v304)
  %v307 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %v306)
  %v308 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %v305)
  %v309 = add nsw i32 %v299, -63
  %v310 = getelementptr inbounds i8, ptr %v48, i32 %v309
  %v312 = load <16 x i32>, ptr %v310, align 1, !tbaa !7
  %v313 = tail call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %v312)
  %v314 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v313)
  %v315 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %v314)
  %v316 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v315)
  %v317 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v315)
  %v318 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v316, i32 168430090)
  %v319 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v317, i32 168430090)
  %v320 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v318, <16 x i32> %v319)
  %v321 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v313)
  %v322 = tail call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %v321)
  %v323 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v322)
  %v324 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v322)
  %v325 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v323, i32 168430090)
  %v326 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v324, i32 168430090)
  %v327 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v325, <16 x i32> %v326)
  %v328 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v307, <32 x i32> %v320)
  %v329 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v308, <32 x i32> %v327)
  %v330 = shufflevector <32 x i32> %v328, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v331 = mul nsw i32 %v296, %v222
  %v332 = add nsw i32 %v331, %v298
  %v333 = add nsw i32 %v332, -64
  %v335 = getelementptr inbounds i32, ptr %v227, i32 %v333
  store <16 x i32> %v330, ptr %v335, align 4, !tbaa !10
  %v337 = shufflevector <32 x i32> %v328, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v338 = add nsw i32 %v332, -48
  %v339 = getelementptr inbounds i32, ptr %v227, i32 %v338
  store <16 x i32> %v337, ptr %v339, align 4, !tbaa !10
  %v341 = shufflevector <32 x i32> %v329, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v342 = add nsw i32 %v332, -32
  %v343 = getelementptr inbounds i32, ptr %v227, i32 %v342
  store <16 x i32> %v341, ptr %v343, align 4, !tbaa !10
  %v345 = shufflevector <32 x i32> %v329, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v346 = add nsw i32 %v332, -16
  %v347 = getelementptr inbounds i32, ptr %v227, i32 %v346
  store <16 x i32> %v345, ptr %v347, align 4, !tbaa !10
  %v349 = add nuw nsw i32 %v294, 1
  %v350 = icmp eq i32 %v349, %v292
  br i1 %v350, label %b19, label %b18

b19:                                              ; preds = %b18, %b17
  %v351 = add nsw i32 %v228, 1
  %v352 = icmp eq i32 %v351, %v50
  br i1 %v352, label %b14, label %b13

b20:                                              ; preds = %b15, %b14
  %v353 = icmp sgt i32 %v17, 0
  br i1 %v353, label %b21, label %b31, !prof !9

b21:                                              ; preds = %b20
  %v354 = ashr i32 %v15, 6
  %v355 = icmp slt i32 %v354, 0
  %v356 = select i1 %v355, i32 0, i32 %v354
  %v357 = icmp sgt i32 %v356, 0
  br i1 %v357, label %b25, label %b27

b22:                                              ; preds = %b25, %b22
  %v358 = phi i32 [ %v442, %b22 ], [ 0, %b25 ]
  %v359 = sub nsw i32 %v525, %v23
  %v360 = mul nsw i32 %v359, %v222
  %v361 = shl nsw i32 %v358, 6
  %v362 = add nsw i32 %v361, %v21
  %v363 = sub nsw i32 %v362, %v27
  %v364 = add nsw i32 %v363, %v360
  %v366 = getelementptr inbounds i32, ptr %v227, i32 %v364
  %v368 = load <16 x i32>, ptr %v366, align 4, !tbaa !10
  %v369 = add nsw i32 %v364, 16
  %v370 = getelementptr inbounds i32, ptr %v227, i32 %v369
  %v372 = load <16 x i32>, ptr %v370, align 4, !tbaa !10
  %v373 = shufflevector <16 x i32> %v368, <16 x i32> %v372, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v374 = add nsw i32 %v359, 1
  %v375 = mul nsw i32 %v374, %v222
  %v376 = add nsw i32 %v363, %v375
  %v377 = getelementptr inbounds i32, ptr %v227, i32 %v376
  %v379 = load <16 x i32>, ptr %v377, align 4, !tbaa !10
  %v380 = add nsw i32 %v376, 16
  %v381 = getelementptr inbounds i32, ptr %v227, i32 %v380
  %v383 = load <16 x i32>, ptr %v381, align 4, !tbaa !10
  %v384 = shufflevector <16 x i32> %v379, <16 x i32> %v383, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v385 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v384)
  %v386 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v384)
  %v387 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v385, i32 168430090)
  %v388 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v386, i32 168430090)
  %v389 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v387, <16 x i32> %v388)
  %v390 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v373, <32 x i32> %v389)
  %v391 = shufflevector <32 x i32> %v390, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v392 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v391, i32 20)
  %v393 = shufflevector <32 x i32> %v390, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v394 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v393, i32 20)
  %v395 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v394, <16 x i32> %v392)
  %v396 = add nsw i32 %v364, 32
  %v397 = getelementptr inbounds i32, ptr %v227, i32 %v396
  %v399 = load <16 x i32>, ptr %v397, align 4, !tbaa !10
  %v400 = add nsw i32 %v364, 48
  %v401 = getelementptr inbounds i32, ptr %v227, i32 %v400
  %v403 = load <16 x i32>, ptr %v401, align 4, !tbaa !10
  %v404 = shufflevector <16 x i32> %v399, <16 x i32> %v403, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v405 = add nsw i32 %v376, 32
  %v406 = getelementptr inbounds i32, ptr %v227, i32 %v405
  %v408 = load <16 x i32>, ptr %v406, align 4, !tbaa !10
  %v409 = add nsw i32 %v376, 48
  %v410 = getelementptr inbounds i32, ptr %v227, i32 %v409
  %v412 = load <16 x i32>, ptr %v410, align 4, !tbaa !10
  %v413 = shufflevector <16 x i32> %v408, <16 x i32> %v412, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v414 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v413)
  %v415 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v413)
  %v416 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v414, i32 168430090)
  %v417 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v415, i32 168430090)
  %v418 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v416, <16 x i32> %v417)
  %v419 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v404, <32 x i32> %v418)
  %v420 = shufflevector <32 x i32> %v419, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v421 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v420, i32 20)
  %v422 = shufflevector <32 x i32> %v419, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v423 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v422, i32 20)
  %v424 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v423, <16 x i32> %v421)
  %v425 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v395)
  %v426 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v395)
  %v427 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %v425, <16 x i32> %v426)
  %v428 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v424)
  %v429 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v424)
  %v430 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %v428, <16 x i32> %v429)
  %v431 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v430, <16 x i32> %v427)
  %v432 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v431)
  %v433 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v431)
  %v434 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v432, <16 x i32> %v433)
  %v435 = mul nsw i32 %v23, %v19
  %v436 = mul nsw i32 %v525, %v19
  %v437 = add i32 %v435, %v21
  %v438 = sub i32 %v436, %v437
  %v439 = add i32 %v438, %v362
  %v440 = getelementptr inbounds i8, ptr %v13, i32 %v439
  store <16 x i32> %v434, ptr %v440, align 1, !tbaa !12
  %v442 = add nuw nsw i32 %v358, 1
  %v443 = icmp eq i32 %v442, %v356
  br i1 %v443, label %b26, label %b22

b23:                                              ; preds = %b26, %b23
  %v444 = phi i32 [ %v521, %b23 ], [ %v356, %b26 ]
  %v445 = sub nsw i32 %v24, %v27
  %v446 = add nsw i32 %v360, %v445
  %v447 = add nsw i32 %v446, -64
  %v448 = getelementptr inbounds i32, ptr %v227, i32 %v447
  %v450 = load <16 x i32>, ptr %v448, align 4, !tbaa !10
  %v451 = add nsw i32 %v446, -48
  %v452 = getelementptr inbounds i32, ptr %v227, i32 %v451
  %v454 = load <16 x i32>, ptr %v452, align 4, !tbaa !10
  %v455 = shufflevector <16 x i32> %v450, <16 x i32> %v454, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v456 = add nsw i32 %v375, %v445
  %v457 = add nsw i32 %v456, -64
  %v458 = getelementptr inbounds i32, ptr %v227, i32 %v457
  %v460 = load <16 x i32>, ptr %v458, align 4, !tbaa !10
  %v461 = add nsw i32 %v456, -48
  %v462 = getelementptr inbounds i32, ptr %v227, i32 %v461
  %v464 = load <16 x i32>, ptr %v462, align 4, !tbaa !10
  %v465 = shufflevector <16 x i32> %v460, <16 x i32> %v464, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v466 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v465)
  %v467 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v465)
  %v468 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v466, i32 168430090)
  %v469 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v467, i32 168430090)
  %v470 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v468, <16 x i32> %v469)
  %v471 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v455, <32 x i32> %v470)
  %v472 = shufflevector <32 x i32> %v471, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v473 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v472, i32 20)
  %v474 = shufflevector <32 x i32> %v471, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v475 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v474, i32 20)
  %v476 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v475, <16 x i32> %v473)
  %v477 = add nsw i32 %v446, -32
  %v478 = getelementptr inbounds i32, ptr %v227, i32 %v477
  %v480 = load <16 x i32>, ptr %v478, align 4, !tbaa !10
  %v481 = add nsw i32 %v446, -16
  %v482 = getelementptr inbounds i32, ptr %v227, i32 %v481
  %v484 = load <16 x i32>, ptr %v482, align 4, !tbaa !10
  %v485 = shufflevector <16 x i32> %v480, <16 x i32> %v484, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v486 = add nsw i32 %v456, -32
  %v487 = getelementptr inbounds i32, ptr %v227, i32 %v486
  %v489 = load <16 x i32>, ptr %v487, align 4, !tbaa !10
  %v490 = add nsw i32 %v456, -16
  %v491 = getelementptr inbounds i32, ptr %v227, i32 %v490
  %v493 = load <16 x i32>, ptr %v491, align 4, !tbaa !10
  %v494 = shufflevector <16 x i32> %v489, <16 x i32> %v493, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v495 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v494)
  %v496 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v494)
  %v497 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v495, i32 168430090)
  %v498 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v496, i32 168430090)
  %v499 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v497, <16 x i32> %v498)
  %v500 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v485, <32 x i32> %v499)
  %v501 = shufflevector <32 x i32> %v500, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v502 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v501, i32 20)
  %v503 = shufflevector <32 x i32> %v500, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v504 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v503, i32 20)
  %v505 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v504, <16 x i32> %v502)
  %v506 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v476)
  %v507 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v476)
  %v508 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %v506, <16 x i32> %v507)
  %v509 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v505)
  %v510 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v505)
  %v511 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %v509, <16 x i32> %v510)
  %v512 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v511, <16 x i32> %v508)
  %v513 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v512)
  %v514 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v512)
  %v515 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v513, <16 x i32> %v514)
  %v516 = add i32 %v15, -64
  %v517 = sub i32 %v516, %v435
  %v518 = add i32 %v517, %v436
  %v519 = getelementptr inbounds i8, ptr %v13, i32 %v518
  store <16 x i32> %v515, ptr %v519, align 1, !tbaa !12
  %v521 = add nuw nsw i32 %v444, 1
  %v522 = icmp eq i32 %v521, %v527
  br i1 %v522, label %b24, label %b23

b24:                                              ; preds = %b26, %b23
  %v523 = add nsw i32 %v525, 1
  %v524 = icmp eq i32 %v523, %v50
  br i1 %v524, label %b32, label %b25

b25:                                              ; preds = %b24, %b21
  %v525 = phi i32 [ %v523, %b24 ], [ %v23, %b21 ]
  br label %b22

b26:                                              ; preds = %b22
  %v526 = add nsw i32 %v15, 63
  %v527 = ashr i32 %v526, 6
  %v528 = icmp slt i32 %v356, %v527
  br i1 %v528, label %b23, label %b24, !prof !9

b27:                                              ; preds = %b21
  %v529 = add nsw i32 %v15, 63
  %v530 = ashr i32 %v529, 6
  %v531 = icmp slt i32 %v356, %v530
  br i1 %v531, label %b29, label %b31

b28:                                              ; preds = %b29, %b28
  %v532 = phi i32 [ %v616, %b28 ], [ %v356, %b29 ]
  %v533 = sub nsw i32 %v618, %v23
  %v534 = mul nsw i32 %v533, %v222
  %v535 = sub nsw i32 %v24, %v27
  %v536 = add nsw i32 %v534, %v535
  %v537 = add nsw i32 %v536, -64
  %v539 = getelementptr inbounds i32, ptr %v227, i32 %v537
  %v541 = load <16 x i32>, ptr %v539, align 4, !tbaa !10
  %v542 = add nsw i32 %v536, -48
  %v543 = getelementptr inbounds i32, ptr %v227, i32 %v542
  %v545 = load <16 x i32>, ptr %v543, align 4, !tbaa !10
  %v546 = shufflevector <16 x i32> %v541, <16 x i32> %v545, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v547 = add nsw i32 %v533, 1
  %v548 = mul nsw i32 %v547, %v222
  %v549 = add nsw i32 %v548, %v535
  %v550 = add nsw i32 %v549, -64
  %v551 = getelementptr inbounds i32, ptr %v227, i32 %v550
  %v553 = load <16 x i32>, ptr %v551, align 4, !tbaa !10
  %v554 = add nsw i32 %v549, -48
  %v555 = getelementptr inbounds i32, ptr %v227, i32 %v554
  %v557 = load <16 x i32>, ptr %v555, align 4, !tbaa !10
  %v558 = shufflevector <16 x i32> %v553, <16 x i32> %v557, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v559 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v558)
  %v560 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v558)
  %v561 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v559, i32 168430090)
  %v562 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v560, i32 168430090)
  %v563 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v561, <16 x i32> %v562)
  %v564 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v546, <32 x i32> %v563)
  %v565 = shufflevector <32 x i32> %v564, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v566 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v565, i32 20)
  %v567 = shufflevector <32 x i32> %v564, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v568 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v567, i32 20)
  %v569 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v568, <16 x i32> %v566)
  %v570 = add nsw i32 %v536, -32
  %v571 = getelementptr inbounds i32, ptr %v227, i32 %v570
  %v573 = load <16 x i32>, ptr %v571, align 4, !tbaa !10
  %v574 = add nsw i32 %v536, -16
  %v575 = getelementptr inbounds i32, ptr %v227, i32 %v574
  %v577 = load <16 x i32>, ptr %v575, align 4, !tbaa !10
  %v578 = shufflevector <16 x i32> %v573, <16 x i32> %v577, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v579 = add nsw i32 %v549, -32
  %v580 = getelementptr inbounds i32, ptr %v227, i32 %v579
  %v582 = load <16 x i32>, ptr %v580, align 4, !tbaa !10
  %v583 = add nsw i32 %v549, -16
  %v584 = getelementptr inbounds i32, ptr %v227, i32 %v583
  %v586 = load <16 x i32>, ptr %v584, align 4, !tbaa !10
  %v587 = shufflevector <16 x i32> %v582, <16 x i32> %v586, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v588 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v587)
  %v589 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v587)
  %v590 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v588, i32 168430090)
  %v591 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %v589, i32 168430090)
  %v592 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v590, <16 x i32> %v591)
  %v593 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v578, <32 x i32> %v592)
  %v594 = shufflevector <32 x i32> %v593, <32 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v595 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v594, i32 20)
  %v596 = shufflevector <32 x i32> %v593, <32 x i32> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %v597 = tail call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %v596, i32 20)
  %v598 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v597, <16 x i32> %v595)
  %v599 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v569)
  %v600 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v569)
  %v601 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %v599, <16 x i32> %v600)
  %v602 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v598)
  %v603 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v598)
  %v604 = tail call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %v602, <16 x i32> %v603)
  %v605 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v604, <16 x i32> %v601)
  %v606 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v605)
  %v607 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v605)
  %v608 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v606, <16 x i32> %v607)
  %v609 = mul nsw i32 %v23, %v19
  %v610 = mul nsw i32 %v618, %v19
  %v611 = add i32 %v15, -64
  %v612 = sub i32 %v611, %v609
  %v613 = add i32 %v612, %v610
  %v614 = getelementptr inbounds i8, ptr %v13, i32 %v613
  store <16 x i32> %v608, ptr %v614, align 1, !tbaa !12
  %v616 = add nuw nsw i32 %v532, 1
  %v617 = icmp eq i32 %v616, %v530
  br i1 %v617, label %b30, label %b28

b29:                                              ; preds = %b30, %b27
  %v618 = phi i32 [ %v619, %b30 ], [ %v23, %b27 ]
  br label %b28

b30:                                              ; preds = %b28
  %v619 = add nsw i32 %v618, 1
  %v620 = icmp eq i32 %v619, %v50
  br i1 %v620, label %b32, label %b29

b31:                                              ; preds = %b27, %b20
  %v621 = icmp eq ptr %v227, null
  br i1 %v621, label %b33, label %b32

b32:                                              ; preds = %b31, %b30, %b24
  tail call void @f3(ptr null, ptr %v227) #2
  br label %b33

b33:                                              ; preds = %b32, %b31
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nobuiltin nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"halide_use_soft_float_abi", i32 0}
!1 = !{i32 2, !"halide_mcpu", !"hexagonv60"}
!2 = !{i32 2, !"halide_mattrs", !"+hvx"}
!3 = !{!"branch_weights", i32 0, i32 1073741824}
!4 = !{!5, !5, i64 0}
!5 = !{!"input", !6}
!6 = !{!"Halide buffer"}
!7 = !{!8, !8, i64 0}
!8 = !{!"constant_exterior", !6}
!9 = !{!"branch_weights", i32 1073741824, i32 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"rows", !6}
!12 = !{!13, !13, i64 0}
!13 = !{!"gaussian11", !6}
