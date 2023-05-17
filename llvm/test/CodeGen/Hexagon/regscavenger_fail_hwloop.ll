; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; This test used to fail with;
;  Assertion `ScavengingFrameIndex >= 0 && "Cannot scavenge register without an
;             emergency spill slot!"' failed.

target triple = "hexagon-unknown-linux-gnu"

; Function Attrs: nounwind
define hidden fastcc void @f0(ptr nocapture %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, ptr nocapture %a5) #0 {
b0:
  %v0 = add i32 %a3, -4
  %v1 = icmp ult i32 %v0, %a1
  %v2 = add i32 %a2, -2
  br i1 %v1, label %b2, label %b1

b1:                                               ; preds = %b0
  %v3 = add i32 %a4, -9
  %v4 = icmp ugt i32 %v2, %v3
  br i1 %v4, label %b2, label %b3

b2:                                               ; preds = %b1, %b0
  %v5 = add nsw i32 %a4, -1
  %v6 = add nsw i32 %a3, -1
  %v7 = add i32 %a2, 1
  %v8 = add i32 %a2, 2
  %v9 = icmp slt i32 %v7, 0
  %v10 = icmp slt i32 %v7, %a4
  %v11 = select i1 %v10, i32 %v7, i32 %v5
  %v12 = select i1 %v9, i32 0, i32 %v11
  %v13 = mul i32 %v12, %a3
  %v14 = add i32 %a2, -1
  %v15 = icmp slt i32 %v2, 0
  %v16 = icmp slt i32 %v2, %a4
  %v17 = select i1 %v16, i32 %v2, i32 %v5
  %v18 = select i1 %v15, i32 0, i32 %v17
  %v19 = mul i32 %v18, %a3
  %v20 = icmp slt i32 %v14, 0
  %v21 = icmp slt i32 %v14, %a4
  %v22 = select i1 %v21, i32 %v14, i32 %v5
  %v23 = select i1 %v20, i32 0, i32 %v22
  %v24 = mul i32 %v23, %a3
  %v25 = icmp slt i32 %a2, 0
  %v26 = icmp slt i32 %a2, %a4
  %v27 = select i1 %v26, i32 %a2, i32 %v5
  %v28 = select i1 %v25, i32 0, i32 %v27
  %v29 = mul i32 %v28, %a3
  %v30 = add i32 %a2, 3
  %v31 = icmp slt i32 %v8, 0
  %v32 = icmp slt i32 %v8, %a4
  %v33 = select i1 %v32, i32 %v8, i32 %v5
  %v34 = select i1 %v31, i32 0, i32 %v33
  %v35 = mul i32 %v34, %a3
  %v36 = icmp slt i32 %v30, 0
  %v37 = icmp slt i32 %v30, %a4
  %v38 = select i1 %v37, i32 %v30, i32 %v5
  %v39 = select i1 %v36, i32 0, i32 %v38
  %v40 = mul i32 %v39, %a3
  %v41 = add i32 %a2, 4
  %v42 = icmp slt i32 %v41, 0
  %v43 = icmp slt i32 %v41, %a4
  %v44 = select i1 %v43, i32 %v41, i32 %v5
  %v45 = select i1 %v42, i32 0, i32 %v44
  %v46 = mul i32 %v45, %a3
  %v47 = add i32 %a2, 5
  %v48 = icmp slt i32 %v47, 0
  %v49 = icmp slt i32 %v47, %a4
  %v50 = select i1 %v49, i32 %v47, i32 %v5
  %v51 = select i1 %v48, i32 0, i32 %v50
  %v52 = mul i32 %v51, %a3
  %v53 = add i32 %a2, 6
  %v54 = icmp slt i32 %v53, 0
  %v55 = icmp slt i32 %v53, %a4
  %v56 = select i1 %v55, i32 %v53, i32 %v5
  %v57 = select i1 %v54, i32 0, i32 %v56
  %v58 = mul i32 %v57, %a3
  br label %b5

b3:                                               ; preds = %b1
  %v59 = mul i32 %a3, %a2
  %v60 = add i32 %v59, %a1
  %v61 = getelementptr inbounds i8, ptr %a5, i32 %v60
  %v62 = shl i32 %a3, 1
  %v63 = sub i32 0, %v62
  %v64 = sub i32 %a3, %v62
  %v65 = add i32 %v64, %a3
  %v66 = add i32 %v65, %a3
  %v67 = add i32 %v66, %a3
  %v68 = add i32 %v67, %a3
  %v69 = add i32 %v68, %a3
  %v70 = add i32 %v69, %a3
  %v71 = add i32 %v70, %a3
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v72 = phi ptr [ %a0, %b3 ], [ %v165, %b4 ]
  %v73 = phi ptr [ %v61, %b3 ], [ %v164, %b4 ]
  %v74 = phi i32 [ 4, %b3 ], [ %v166, %b4 ]
  %v75 = getelementptr inbounds i8, ptr %v73, i32 %v63
  %v76 = load i8, ptr %v75, align 1, !tbaa !0
  %v77 = zext i8 %v76 to i32
  %v78 = getelementptr inbounds i8, ptr %v73, i32 %v64
  %v79 = load i8, ptr %v78, align 1, !tbaa !0
  %v80 = zext i8 %v79 to i32
  %v81 = load i8, ptr %v73, align 1, !tbaa !0
  %v82 = zext i8 %v81 to i32
  %v83 = getelementptr inbounds i8, ptr %v73, i32 %v66
  %v84 = load i8, ptr %v83, align 1, !tbaa !0
  %v85 = zext i8 %v84 to i32
  %v86 = getelementptr inbounds i8, ptr %v73, i32 %v67
  %v87 = load i8, ptr %v86, align 1, !tbaa !0
  %v88 = zext i8 %v87 to i32
  %v89 = getelementptr inbounds i8, ptr %v73, i32 %v68
  %v90 = load i8, ptr %v89, align 1, !tbaa !0
  %v91 = zext i8 %v90 to i32
  %v92 = getelementptr inbounds i8, ptr %v73, i32 %v69
  %v93 = load i8, ptr %v92, align 1, !tbaa !0
  %v94 = zext i8 %v93 to i32
  %v95 = getelementptr inbounds i8, ptr %v73, i32 %v70
  %v96 = load i8, ptr %v95, align 1, !tbaa !0
  %v97 = zext i8 %v96 to i32
  %v98 = getelementptr inbounds i8, ptr %v73, i32 %v71
  %v99 = load i8, ptr %v98, align 1, !tbaa !0
  %v100 = zext i8 %v99 to i32
  %v101 = add nsw i32 %v88, %v80
  %v102 = mul i32 %v101, -5
  %v103 = add nsw i32 %v85, %v82
  %v104 = mul nsw i32 %v103, 20
  %v105 = add i32 %v77, 16
  %v106 = add i32 %v105, %v104
  %v107 = add i32 %v106, %v91
  %v108 = add i32 %v107, %v102
  %v109 = ashr i32 %v108, 5
  %v110 = and i32 %v109, 256
  %v111 = icmp ne i32 %v110, 0
  %v112 = lshr i32 %v108, 31
  %v113 = add i32 %v112, 255
  %v114 = select i1 %v111, i32 %v113, i32 %v109
  %v115 = trunc i32 %v114 to i8
  store i8 %v115, ptr %v72, align 1, !tbaa !0
  %v116 = add nsw i32 %v91, %v82
  %v117 = mul i32 %v116, -5
  %v118 = add nsw i32 %v88, %v85
  %v119 = mul nsw i32 %v118, 20
  %v120 = add i32 %v80, 16
  %v121 = add i32 %v120, %v119
  %v122 = add i32 %v121, %v94
  %v123 = add i32 %v122, %v117
  %v124 = ashr i32 %v123, 5
  %v125 = and i32 %v124, 256
  %v126 = icmp ne i32 %v125, 0
  %v127 = lshr i32 %v123, 31
  %v128 = add i32 %v127, 255
  %v129 = select i1 %v126, i32 %v128, i32 %v124
  %v130 = trunc i32 %v129 to i8
  %v131 = getelementptr inbounds i8, ptr %v72, i32 4
  store i8 %v130, ptr %v131, align 1, !tbaa !0
  %v132 = add nsw i32 %v94, %v85
  %v133 = mul i32 %v132, -5
  %v134 = add nsw i32 %v91, %v88
  %v135 = mul nsw i32 %v134, 20
  %v136 = add i32 %v82, 16
  %v137 = add i32 %v136, %v135
  %v138 = add i32 %v137, %v97
  %v139 = add i32 %v138, %v133
  %v140 = ashr i32 %v139, 5
  %v141 = and i32 %v140, 256
  %v142 = icmp ne i32 %v141, 0
  %v143 = lshr i32 %v139, 31
  %v144 = add i32 %v143, 255
  %v145 = select i1 %v142, i32 %v144, i32 %v140
  %v146 = trunc i32 %v145 to i8
  %v147 = getelementptr inbounds i8, ptr %v72, i32 8
  store i8 %v146, ptr %v147, align 1, !tbaa !0
  %v148 = add nsw i32 %v97, %v88
  %v149 = mul i32 %v148, -5
  %v150 = add nsw i32 %v94, %v91
  %v151 = mul nsw i32 %v150, 20
  %v152 = add i32 %v85, 16
  %v153 = add i32 %v152, %v151
  %v154 = add i32 %v153, %v100
  %v155 = add i32 %v154, %v149
  %v156 = ashr i32 %v155, 5
  %v157 = and i32 %v156, 256
  %v158 = icmp ne i32 %v157, 0
  %v159 = lshr i32 %v155, 31
  %v160 = add i32 %v159, 255
  %v161 = select i1 %v158, i32 %v160, i32 %v156
  %v162 = trunc i32 %v161 to i8
  %v163 = getelementptr inbounds i8, ptr %v72, i32 12
  store i8 %v162, ptr %v163, align 1, !tbaa !0
  %v164 = getelementptr inbounds i8, ptr %v73, i32 1
  %v165 = getelementptr inbounds i8, ptr %v72, i32 1
  %v166 = add i32 %v74, -1
  %v167 = icmp eq i32 %v166, 0
  br i1 %v167, label %b7, label %b4

b5:                                               ; preds = %b5, %b2
  %v168 = phi ptr [ %a0, %b2 ], [ %v312, %b5 ]
  %v169 = phi i32 [ 0, %b2 ], [ %v313, %b5 ]
  %v170 = add i32 %v169, %a1
  %v171 = icmp slt i32 %v170, 0
  %v172 = icmp slt i32 %v170, %a3
  %v173 = select i1 %v172, i32 %v170, i32 %v6
  %v174 = select i1 %v171, i32 0, i32 %v173
  %v175 = add i32 %v19, %v174
  %v176 = getelementptr inbounds i8, ptr %a5, i32 %v175
  %v177 = load i8, ptr %v176, align 1, !tbaa !0
  %v178 = zext i8 %v177 to i32
  %v179 = add i32 %v24, %v174
  %v180 = getelementptr inbounds i8, ptr %a5, i32 %v179
  %v181 = load i8, ptr %v180, align 1, !tbaa !0
  %v182 = zext i8 %v181 to i32
  %v183 = mul nsw i32 %v182, -5
  %v184 = add nsw i32 %v183, %v178
  %v185 = add i32 %v29, %v174
  %v186 = getelementptr inbounds i8, ptr %a5, i32 %v185
  %v187 = load i8, ptr %v186, align 1, !tbaa !0
  %v188 = zext i8 %v187 to i32
  %v189 = mul nsw i32 %v188, 20
  %v190 = add nsw i32 %v189, %v184
  %v191 = add i32 %v13, %v174
  %v192 = getelementptr inbounds i8, ptr %a5, i32 %v191
  %v193 = load i8, ptr %v192, align 1, !tbaa !0
  %v194 = zext i8 %v193 to i32
  %v195 = mul nsw i32 %v194, 20
  %v196 = add nsw i32 %v195, %v190
  %v197 = add i32 %v35, %v174
  %v198 = getelementptr inbounds i8, ptr %a5, i32 %v197
  %v199 = load i8, ptr %v198, align 1, !tbaa !0
  %v200 = zext i8 %v199 to i32
  %v201 = mul nsw i32 %v200, -5
  %v202 = add nsw i32 %v201, %v196
  %v203 = add i32 %v40, %v174
  %v204 = getelementptr inbounds i8, ptr %a5, i32 %v203
  %v205 = load i8, ptr %v204, align 1, !tbaa !0
  %v206 = zext i8 %v205 to i32
  %v207 = add nsw i32 %v206, %v202
  %v208 = add nsw i32 %v207, 16
  %v209 = ashr i32 %v208, 5
  %v210 = and i32 %v209, 256
  %v211 = icmp ne i32 %v210, 0
  %v212 = lshr i32 %v208, 31
  %v213 = add i32 %v212, 255
  %v214 = select i1 %v211, i32 %v213, i32 %v209
  %v215 = trunc i32 %v214 to i8
  store i8 %v215, ptr %v168, align 1, !tbaa !0
  %v216 = getelementptr inbounds i8, ptr %v168, i32 4
  %v217 = load i8, ptr %v180, align 1, !tbaa !0
  %v218 = zext i8 %v217 to i32
  %v219 = load i8, ptr %v186, align 1, !tbaa !0
  %v220 = zext i8 %v219 to i32
  %v221 = mul nsw i32 %v220, -5
  %v222 = add nsw i32 %v221, %v218
  %v223 = load i8, ptr %v192, align 1, !tbaa !0
  %v224 = zext i8 %v223 to i32
  %v225 = mul nsw i32 %v224, 20
  %v226 = add nsw i32 %v225, %v222
  %v227 = load i8, ptr %v198, align 1, !tbaa !0
  %v228 = zext i8 %v227 to i32
  %v229 = mul nsw i32 %v228, 20
  %v230 = add nsw i32 %v229, %v226
  %v231 = load i8, ptr %v204, align 1, !tbaa !0
  %v232 = zext i8 %v231 to i32
  %v233 = mul nsw i32 %v232, -5
  %v234 = add nsw i32 %v233, %v230
  %v235 = add i32 %v46, %v174
  %v236 = getelementptr inbounds i8, ptr %a5, i32 %v235
  %v237 = load i8, ptr %v236, align 1, !tbaa !0
  %v238 = zext i8 %v237 to i32
  %v239 = add nsw i32 %v238, %v234
  %v240 = add nsw i32 %v239, 16
  %v241 = ashr i32 %v240, 5
  %v242 = and i32 %v241, 256
  %v243 = icmp ne i32 %v242, 0
  %v244 = lshr i32 %v240, 31
  %v245 = add i32 %v244, 255
  %v246 = select i1 %v243, i32 %v245, i32 %v241
  %v247 = trunc i32 %v246 to i8
  store i8 %v247, ptr %v216, align 1, !tbaa !0
  %v248 = getelementptr inbounds i8, ptr %v168, i32 8
  %v249 = load i8, ptr %v186, align 1, !tbaa !0
  %v250 = zext i8 %v249 to i32
  %v251 = load i8, ptr %v192, align 1, !tbaa !0
  %v252 = zext i8 %v251 to i32
  %v253 = mul nsw i32 %v252, -5
  %v254 = add nsw i32 %v253, %v250
  %v255 = load i8, ptr %v198, align 1, !tbaa !0
  %v256 = zext i8 %v255 to i32
  %v257 = mul nsw i32 %v256, 20
  %v258 = add nsw i32 %v257, %v254
  %v259 = load i8, ptr %v204, align 1, !tbaa !0
  %v260 = zext i8 %v259 to i32
  %v261 = mul nsw i32 %v260, 20
  %v262 = add nsw i32 %v261, %v258
  %v263 = load i8, ptr %v236, align 1, !tbaa !0
  %v264 = zext i8 %v263 to i32
  %v265 = mul nsw i32 %v264, -5
  %v266 = add nsw i32 %v265, %v262
  %v267 = add i32 %v52, %v174
  %v268 = getelementptr inbounds i8, ptr %a5, i32 %v267
  %v269 = load i8, ptr %v268, align 1, !tbaa !0
  %v270 = zext i8 %v269 to i32
  %v271 = add nsw i32 %v270, %v266
  %v272 = add nsw i32 %v271, 16
  %v273 = ashr i32 %v272, 5
  %v274 = and i32 %v273, 256
  %v275 = icmp ne i32 %v274, 0
  %v276 = lshr i32 %v272, 31
  %v277 = add i32 %v276, 255
  %v278 = select i1 %v275, i32 %v277, i32 %v273
  %v279 = trunc i32 %v278 to i8
  store i8 %v279, ptr %v248, align 1, !tbaa !0
  %v280 = getelementptr inbounds i8, ptr %v168, i32 12
  %v281 = load i8, ptr %v192, align 1, !tbaa !0
  %v282 = zext i8 %v281 to i32
  %v283 = load i8, ptr %v198, align 1, !tbaa !0
  %v284 = zext i8 %v283 to i32
  %v285 = mul nsw i32 %v284, -5
  %v286 = add nsw i32 %v285, %v282
  %v287 = load i8, ptr %v204, align 1, !tbaa !0
  %v288 = zext i8 %v287 to i32
  %v289 = mul nsw i32 %v288, 20
  %v290 = add nsw i32 %v289, %v286
  %v291 = load i8, ptr %v236, align 1, !tbaa !0
  %v292 = zext i8 %v291 to i32
  %v293 = mul nsw i32 %v292, 20
  %v294 = add nsw i32 %v293, %v290
  %v295 = load i8, ptr %v268, align 1, !tbaa !0
  %v296 = zext i8 %v295 to i32
  %v297 = mul nsw i32 %v296, -5
  %v298 = add nsw i32 %v297, %v294
  %v299 = add i32 %v58, %v174
  %v300 = getelementptr inbounds i8, ptr %a5, i32 %v299
  %v301 = load i8, ptr %v300, align 1, !tbaa !0
  %v302 = zext i8 %v301 to i32
  %v303 = add nsw i32 %v302, %v298
  %v304 = add nsw i32 %v303, 16
  %v305 = ashr i32 %v304, 5
  %v306 = and i32 %v305, 256
  %v307 = icmp ne i32 %v306, 0
  %v308 = lshr i32 %v304, 31
  %v309 = add i32 %v308, 255
  %v310 = select i1 %v307, i32 %v309, i32 %v305
  %v311 = trunc i32 %v310 to i8
  store i8 %v311, ptr %v280, align 1, !tbaa !0
  %v312 = getelementptr inbounds i8, ptr %v168, i32 1
  %v313 = add i32 %v169, 1
  %v314 = icmp eq i32 %v313, 4
  br i1 %v314, label %b6, label %b5

b6:                                               ; preds = %b5
  br label %b8

b7:                                               ; preds = %b4
  br label %b8

b8:                                               ; preds = %b7, %b6
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
