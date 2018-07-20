; RUN: llc  -march=hexagon < %s | FileCheck %s
; This test is no longer connected to HRC.

target triple = "hexagon"

%s.0 = type { %s.1*, %s.2*, %s.3*, i16*, i32*, i8, i8, i8, i8, i8, i8, i16, i16, i16, i32, i32, i32, i32, i16, i8, i8, i8, i8, float, float, float, float, float, float, float, float, float, float, float, [4 x %s.7], [4 x %s.7], [20 x %s.7], [104 x %s.7], [20 x i32], [257 x %s.8], %s.9 }
%s.1 = type { i16, i8, i16, i8, i8, i8, i8, i8 }
%s.2 = type { i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, [20 x i16], i8, i16 }
%s.3 = type { i8, i8, i8, i8, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i32, i32, i32, [2 x [2 x i32]], %s.4 }
%s.4 = type { %s.5, [976 x i8] }
%s.5 = type { %s.6 }
%s.6 = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 }
%s.7 = type { i64 }
%s.8 = type { i32, i32 }
%s.9 = type { %s.10, [1960 x i8] }
%s.10 = type { i64, i64, i64, i64, i64, i64, i64, [104 x %s.11], [104 x float] }
%s.11 = type { i64, i64 }
%s.12 = type { float, float }

; CHECK: .type   f0,@function
; This allocframe argument value may change, but typically should remain
; in the 250-280 range. This test was introduced to test a change that
; reduced stack usage from around 568 bytes to 280 bytes.
; After r308350 the stack size is ~300.
; CHECK: allocframe(r29,#304):raw
define void @f0(%s.0* %a0, %s.11* %a1, %s.12* %a2) #0 {
b0:
  %v0 = alloca %s.0*, align 4
  %v1 = alloca %s.11*, align 4
  %v2 = alloca %s.12*, align 4
  %v3 = alloca float, align 4
  %v4 = alloca float, align 4
  %v5 = alloca float, align 4
  %v6 = alloca float, align 4
  %v7 = alloca float, align 4
  %v8 = alloca float, align 4
  %v9 = alloca float, align 4
  %v10 = alloca float, align 4
  %v11 = alloca float, align 4
  %v12 = alloca float, align 4
  %v13 = alloca double, align 8
  %v14 = alloca double, align 8
  %v15 = alloca double, align 8
  %v16 = alloca double, align 8
  %v17 = alloca double, align 8
  %v18 = alloca double, align 8
  %v19 = alloca double, align 8
  %v20 = alloca double, align 8
  %v21 = alloca double, align 8
  %v22 = alloca double, align 8
  %v23 = alloca double, align 8
  %v24 = alloca double, align 8
  %v25 = alloca double, align 8
  %v26 = alloca double, align 8
  %v27 = alloca double, align 8
  %v28 = alloca double, align 8
  %v29 = alloca double, align 8
  %v30 = alloca double, align 8
  %v31 = alloca double, align 8
  %v32 = alloca double, align 8
  %v33 = alloca double, align 8
  store %s.0* %a0, %s.0** %v0, align 4
  store %s.11* %a1, %s.11** %v1, align 4
  store %s.12* %a2, %s.12** %v2, align 4
  store double 1.000000e+00, double* %v32, align 8
  %v34 = load %s.11*, %s.11** %v1, align 4
  %v35 = getelementptr inbounds %s.11, %s.11* %v34, i32 0
  %v36 = getelementptr inbounds %s.11, %s.11* %v35, i32 0, i32 0
  %v37 = load i64, i64* %v36, align 8
  %v38 = sitofp i64 %v37 to double
  %v39 = load double, double* %v32, align 8
  %v40 = fmul double %v38, %v39
  store double %v40, double* %v13, align 8
  %v41 = load %s.11*, %s.11** %v1, align 4
  %v42 = getelementptr inbounds %s.11, %s.11* %v41, i32 1
  %v43 = getelementptr inbounds %s.11, %s.11* %v42, i32 0, i32 0
  %v44 = load i64, i64* %v43, align 8
  %v45 = sitofp i64 %v44 to double
  %v46 = load double, double* %v32, align 8
  %v47 = fmul double %v45, %v46
  store double %v47, double* %v14, align 8
  %v48 = load %s.11*, %s.11** %v1, align 4
  %v49 = getelementptr inbounds %s.11, %s.11* %v48, i32 1
  %v50 = getelementptr inbounds %s.11, %s.11* %v49, i32 0, i32 1
  %v51 = load i64, i64* %v50, align 8
  %v52 = sitofp i64 %v51 to double
  %v53 = load double, double* %v32, align 8
  %v54 = fmul double %v52, %v53
  store double %v54, double* %v15, align 8
  %v55 = load %s.11*, %s.11** %v1, align 4
  %v56 = getelementptr inbounds %s.11, %s.11* %v55, i32 2
  %v57 = getelementptr inbounds %s.11, %s.11* %v56, i32 0, i32 0
  %v58 = load i64, i64* %v57, align 8
  %v59 = sitofp i64 %v58 to double
  %v60 = load double, double* %v32, align 8
  %v61 = fmul double %v59, %v60
  store double %v61, double* %v16, align 8
  %v62 = load %s.11*, %s.11** %v1, align 4
  %v63 = getelementptr inbounds %s.11, %s.11* %v62, i32 2
  %v64 = getelementptr inbounds %s.11, %s.11* %v63, i32 0, i32 1
  %v65 = load i64, i64* %v64, align 8
  %v66 = sitofp i64 %v65 to double
  %v67 = load double, double* %v32, align 8
  %v68 = fmul double %v66, %v67
  store double %v68, double* %v17, align 8
  %v69 = load %s.11*, %s.11** %v1, align 4
  %v70 = getelementptr inbounds %s.11, %s.11* %v69, i32 3
  %v71 = getelementptr inbounds %s.11, %s.11* %v70, i32 0, i32 0
  %v72 = load i64, i64* %v71, align 8
  %v73 = sitofp i64 %v72 to double
  %v74 = load double, double* %v32, align 8
  %v75 = fmul double %v73, %v74
  store double %v75, double* %v18, align 8
  %v76 = load %s.11*, %s.11** %v1, align 4
  %v77 = getelementptr inbounds %s.11, %s.11* %v76, i32 3
  %v78 = getelementptr inbounds %s.11, %s.11* %v77, i32 0, i32 1
  %v79 = load i64, i64* %v78, align 8
  %v80 = sitofp i64 %v79 to double
  %v81 = load double, double* %v32, align 8
  %v82 = fmul double %v80, %v81
  store double %v82, double* %v19, align 8
  %v83 = load double, double* %v13, align 8
  %v84 = load double, double* %v13, align 8
  %v85 = fmul double %v83, %v84
  %v86 = load double, double* %v14, align 8
  %v87 = load double, double* %v14, align 8
  %v88 = fmul double %v86, %v87
  %v89 = fsub double %v85, %v88
  %v90 = load double, double* %v15, align 8
  %v91 = load double, double* %v15, align 8
  %v92 = fmul double %v90, %v91
  %v93 = fsub double %v89, %v92
  store double %v93, double* %v20, align 8
  %v94 = load double, double* %v13, align 8
  %v95 = load double, double* %v14, align 8
  %v96 = fmul double %v94, %v95
  %v97 = load double, double* %v16, align 8
  %v98 = load double, double* %v14, align 8
  %v99 = fmul double %v97, %v98
  %v100 = fsub double %v96, %v99
  %v101 = load double, double* %v17, align 8
  %v102 = load double, double* %v15, align 8
  %v103 = fmul double %v101, %v102
  %v104 = fsub double %v100, %v103
  store double %v104, double* %v21, align 8
  %v105 = load double, double* %v13, align 8
  %v106 = load double, double* %v15, align 8
  %v107 = fmul double %v105, %v106
  %v108 = load double, double* %v16, align 8
  %v109 = load double, double* %v15, align 8
  %v110 = fmul double %v108, %v109
  %v111 = fadd double %v107, %v110
  %v112 = load double, double* %v17, align 8
  %v113 = load double, double* %v14, align 8
  %v114 = fmul double %v112, %v113
  %v115 = fsub double %v111, %v114
  store double %v115, double* %v22, align 8
  %v116 = load double, double* %v13, align 8
  %v117 = load double, double* %v16, align 8
  %v118 = fmul double %v116, %v117
  %v119 = load double, double* %v18, align 8
  %v120 = load double, double* %v14, align 8
  %v121 = fmul double %v119, %v120
  %v122 = fsub double %v118, %v121
  %v123 = load double, double* %v19, align 8
  %v124 = load double, double* %v15, align 8
  %v125 = fmul double %v123, %v124
  %v126 = fsub double %v122, %v125
  store double %v126, double* %v23, align 8
  %v127 = load double, double* %v13, align 8
  %v128 = load double, double* %v17, align 8
  %v129 = fmul double %v127, %v128
  %v130 = load double, double* %v18, align 8
  %v131 = load double, double* %v15, align 8
  %v132 = fmul double %v130, %v131
  %v133 = fadd double %v129, %v132
  %v134 = load double, double* %v19, align 8
  %v135 = load double, double* %v14, align 8
  %v136 = fmul double %v134, %v135
  %v137 = fsub double %v133, %v136
  store double %v137, double* %v24, align 8
  %v138 = load double, double* %v14, align 8
  %v139 = load double, double* %v14, align 8
  %v140 = fmul double %v138, %v139
  %v141 = load double, double* %v15, align 8
  %v142 = load double, double* %v15, align 8
  %v143 = fmul double %v141, %v142
  %v144 = fsub double %v140, %v143
  %v145 = load double, double* %v16, align 8
  %v146 = load double, double* %v13, align 8
  %v147 = fmul double %v145, %v146
  %v148 = fsub double %v144, %v147
  store double %v148, double* %v25, align 8
  %v149 = load double, double* %v14, align 8
  %v150 = load double, double* %v15, align 8
  %v151 = fmul double %v149, %v150
  %v152 = fmul double %v151, 2.000000e+00
  %v153 = load double, double* %v17, align 8
  %v154 = load double, double* %v13, align 8
  %v155 = fmul double %v153, %v154
  %v156 = fsub double %v152, %v155
  store double %v156, double* %v26, align 8
  %v157 = load double, double* %v14, align 8
  %v158 = load double, double* %v16, align 8
  %v159 = fmul double %v157, %v158
  %v160 = load double, double* %v15, align 8
  %v161 = load double, double* %v17, align 8
  %v162 = fmul double %v160, %v161
  %v163 = fsub double %v159, %v162
  %v164 = load double, double* %v18, align 8
  %v165 = load double, double* %v13, align 8
  %v166 = fmul double %v164, %v165
  %v167 = fsub double %v163, %v166
  store double %v167, double* %v27, align 8
  %v168 = load double, double* %v14, align 8
  %v169 = load double, double* %v17, align 8
  %v170 = fmul double %v168, %v169
  %v171 = load double, double* %v15, align 8
  %v172 = load double, double* %v16, align 8
  %v173 = fmul double %v171, %v172
  %v174 = fadd double %v170, %v173
  %v175 = load double, double* %v19, align 8
  %v176 = load double, double* %v13, align 8
  %v177 = fmul double %v175, %v176
  %v178 = fsub double %v174, %v177
  store double %v178, double* %v28, align 8
  %v179 = load double, double* %v16, align 8
  %v180 = load double, double* %v16, align 8
  %v181 = fmul double %v179, %v180
  %v182 = load double, double* %v17, align 8
  %v183 = load double, double* %v17, align 8
  %v184 = fmul double %v182, %v183
  %v185 = fsub double %v181, %v184
  %v186 = load double, double* %v18, align 8
  %v187 = load double, double* %v14, align 8
  %v188 = fmul double %v186, %v187
  %v189 = fsub double %v185, %v188
  %v190 = load double, double* %v19, align 8
  %v191 = load double, double* %v15, align 8
  %v192 = fmul double %v190, %v191
  %v193 = fadd double %v189, %v192
  store double %v193, double* %v29, align 8
  %v194 = load double, double* %v16, align 8
  %v195 = load double, double* %v17, align 8
  %v196 = fmul double %v194, %v195
  %v197 = fmul double %v196, 2.000000e+00
  %v198 = load double, double* %v18, align 8
  %v199 = load double, double* %v15, align 8
  %v200 = fmul double %v198, %v199
  %v201 = fsub double %v197, %v200
  %v202 = load double, double* %v19, align 8
  %v203 = load double, double* %v14, align 8
  %v204 = fmul double %v202, %v203
  %v205 = fsub double %v201, %v204
  store double %v205, double* %v30, align 8
  %v206 = load double, double* %v20, align 8
  %v207 = load double, double* %v20, align 8
  %v208 = fmul double %v206, %v207
  %v209 = load double, double* %v21, align 8
  %v210 = load double, double* %v21, align 8
  %v211 = fmul double %v209, %v210
  %v212 = fsub double %v208, %v211
  %v213 = load double, double* %v22, align 8
  %v214 = load double, double* %v22, align 8
  %v215 = fmul double %v213, %v214
  %v216 = fsub double %v212, %v215
  %v217 = load double, double* %v23, align 8
  %v218 = load double, double* %v25, align 8
  %v219 = fmul double %v217, %v218
  %v220 = fmul double %v219, 2.000000e+00
  %v221 = fadd double %v216, %v220
  %v222 = load double, double* %v24, align 8
  %v223 = load double, double* %v26, align 8
  %v224 = fmul double %v222, %v223
  %v225 = fmul double %v224, 2.000000e+00
  %v226 = fadd double %v221, %v225
  %v227 = load double, double* %v27, align 8
  %v228 = load double, double* %v27, align 8
  %v229 = fmul double %v227, %v228
  %v230 = fsub double %v226, %v229
  %v231 = load double, double* %v28, align 8
  %v232 = load double, double* %v28, align 8
  %v233 = fmul double %v231, %v232
  %v234 = fsub double %v230, %v233
  %v235 = load double, double* %v29, align 8
  %v236 = load double, double* %v29, align 8
  %v237 = fmul double %v235, %v236
  %v238 = fadd double %v234, %v237
  %v239 = load double, double* %v30, align 8
  %v240 = load double, double* %v30, align 8
  %v241 = fmul double %v239, %v240
  %v242 = fadd double %v238, %v241
  store double %v242, double* %v31, align 8
  %v243 = load double, double* %v31, align 8
  %v244 = call double @f1(double %v243) #1
  %v245 = load double, double* %v32, align 8
  %v246 = fcmp olt double %v244, %v245
  br i1 %v246, label %b1, label %b2

b1:                                               ; preds = %b0
  %v247 = load %s.0*, %s.0** %v0, align 4
  %v248 = getelementptr inbounds %s.0, %s.0* %v247, i32 0, i32 2
  %v249 = load %s.3*, %s.3** %v248, align 4
  %v250 = getelementptr inbounds %s.3, %s.3* %v249, i32 0, i32 0
  store i8 3, i8* %v250, align 1
  br label %b3

b2:                                               ; preds = %b0
  %v251 = load double, double* %v32, align 8
  %v252 = load double, double* %v31, align 8
  %v253 = fdiv double %v251, %v252
  store double %v253, double* %v32, align 8
  %v254 = load double, double* %v13, align 8
  %v255 = load double, double* %v20, align 8
  %v256 = fmul double %v254, %v255
  %v257 = load double, double* %v14, align 8
  %v258 = load double, double* %v21, align 8
  %v259 = fmul double %v257, %v258
  %v260 = fsub double %v256, %v259
  %v261 = load double, double* %v15, align 8
  %v262 = load double, double* %v22, align 8
  %v263 = fmul double %v261, %v262
  %v264 = fsub double %v260, %v263
  %v265 = load double, double* %v16, align 8
  %v266 = load double, double* %v25, align 8
  %v267 = fmul double %v265, %v266
  %v268 = fadd double %v264, %v267
  %v269 = load double, double* %v17, align 8
  %v270 = load double, double* %v26, align 8
  %v271 = fmul double %v269, %v270
  %v272 = fadd double %v268, %v271
  store double %v272, double* %v33, align 8
  %v273 = load double, double* %v33, align 8
  %v274 = load double, double* %v32, align 8
  %v275 = fmul double %v273, %v274
  %v276 = fptrunc double %v275 to float
  store float %v276, float* %v3, align 4
  %v277 = load double, double* %v14, align 8
  %v278 = fsub double -0.000000e+00, %v277
  %v279 = load double, double* %v20, align 8
  %v280 = fmul double %v278, %v279
  %v281 = load double, double* %v16, align 8
  %v282 = load double, double* %v21, align 8
  %v283 = fmul double %v281, %v282
  %v284 = fadd double %v280, %v283
  %v285 = load double, double* %v17, align 8
  %v286 = load double, double* %v22, align 8
  %v287 = fmul double %v285, %v286
  %v288 = fadd double %v284, %v287
  %v289 = load double, double* %v18, align 8
  %v290 = load double, double* %v25, align 8
  %v291 = fmul double %v289, %v290
  %v292 = fsub double %v288, %v291
  %v293 = load double, double* %v19, align 8
  %v294 = load double, double* %v26, align 8
  %v295 = fmul double %v293, %v294
  %v296 = fsub double %v292, %v295
  store double %v296, double* %v33, align 8
  %v297 = load double, double* %v33, align 8
  %v298 = load double, double* %v32, align 8
  %v299 = fmul double %v297, %v298
  %v300 = fptrunc double %v299 to float
  store float %v300, float* %v4, align 4
  %v301 = load double, double* %v15, align 8
  %v302 = fsub double -0.000000e+00, %v301
  %v303 = load double, double* %v20, align 8
  %v304 = fmul double %v302, %v303
  %v305 = load double, double* %v16, align 8
  %v306 = load double, double* %v22, align 8
  %v307 = fmul double %v305, %v306
  %v308 = fsub double %v304, %v307
  %v309 = load double, double* %v17, align 8
  %v310 = load double, double* %v21, align 8
  %v311 = fmul double %v309, %v310
  %v312 = fadd double %v308, %v311
  %v313 = load double, double* %v18, align 8
  %v314 = load double, double* %v26, align 8
  %v315 = fmul double %v313, %v314
  %v316 = fadd double %v312, %v315
  %v317 = load double, double* %v19, align 8
  %v318 = load double, double* %v25, align 8
  %v319 = fmul double %v317, %v318
  %v320 = fsub double %v316, %v319
  store double %v320, double* %v33, align 8
  %v321 = load double, double* %v33, align 8
  %v322 = load double, double* %v32, align 8
  %v323 = fmul double %v321, %v322
  %v324 = fptrunc double %v323 to float
  store float %v324, float* %v5, align 4
  %v325 = load double, double* %v16, align 8
  %v326 = load double, double* %v29, align 8
  %v327 = fmul double %v325, %v326
  %v328 = load double, double* %v17, align 8
  %v329 = load double, double* %v30, align 8
  %v330 = fmul double %v328, %v329
  %v331 = fadd double %v327, %v330
  %v332 = load double, double* %v14, align 8
  %v333 = load double, double* %v27, align 8
  %v334 = fmul double %v332, %v333
  %v335 = fsub double %v331, %v334
  %v336 = load double, double* %v15, align 8
  %v337 = load double, double* %v28, align 8
  %v338 = fmul double %v336, %v337
  %v339 = fsub double %v335, %v338
  %v340 = load double, double* %v13, align 8
  %v341 = load double, double* %v25, align 8
  %v342 = fmul double %v340, %v341
  %v343 = fadd double %v339, %v342
  store double %v343, double* %v33, align 8
  %v344 = load double, double* %v33, align 8
  %v345 = load double, double* %v32, align 8
  %v346 = fmul double %v344, %v345
  %v347 = fptrunc double %v346 to float
  store float %v347, float* %v6, align 4
  %v348 = load double, double* %v16, align 8
  %v349 = load double, double* %v30, align 8
  %v350 = fmul double %v348, %v349
  %v351 = load double, double* %v17, align 8
  %v352 = load double, double* %v29, align 8
  %v353 = fmul double %v351, %v352
  %v354 = fsub double %v350, %v353
  %v355 = load double, double* %v14, align 8
  %v356 = load double, double* %v28, align 8
  %v357 = fmul double %v355, %v356
  %v358 = fsub double %v354, %v357
  %v359 = load double, double* %v15, align 8
  %v360 = load double, double* %v27, align 8
  %v361 = fmul double %v359, %v360
  %v362 = fadd double %v358, %v361
  %v363 = load double, double* %v13, align 8
  %v364 = load double, double* %v26, align 8
  %v365 = fmul double %v363, %v364
  %v366 = fadd double %v362, %v365
  store double %v366, double* %v33, align 8
  %v367 = load double, double* %v33, align 8
  %v368 = load double, double* %v32, align 8
  %v369 = fmul double %v367, %v368
  %v370 = fptrunc double %v369 to float
  store float %v370, float* %v7, align 4
  %v371 = load double, double* %v14, align 8
  %v372 = fsub double -0.000000e+00, %v371
  %v373 = load double, double* %v29, align 8
  %v374 = fmul double %v372, %v373
  %v375 = load double, double* %v15, align 8
  %v376 = load double, double* %v30, align 8
  %v377 = fmul double %v375, %v376
  %v378 = fsub double %v374, %v377
  %v379 = load double, double* %v13, align 8
  %v380 = load double, double* %v27, align 8
  %v381 = fmul double %v379, %v380
  %v382 = fadd double %v378, %v381
  %v383 = load double, double* %v14, align 8
  %v384 = load double, double* %v25, align 8
  %v385 = fmul double %v383, %v384
  %v386 = fsub double %v382, %v385
  %v387 = load double, double* %v15, align 8
  %v388 = load double, double* %v26, align 8
  %v389 = fmul double %v387, %v388
  %v390 = fadd double %v386, %v389
  store double %v390, double* %v33, align 8
  %v391 = load double, double* %v33, align 8
  %v392 = load double, double* %v32, align 8
  %v393 = fmul double %v391, %v392
  %v394 = fptrunc double %v393 to float
  store float %v394, float* %v8, align 4
  %v395 = load double, double* %v14, align 8
  %v396 = fsub double -0.000000e+00, %v395
  %v397 = load double, double* %v30, align 8
  %v398 = fmul double %v396, %v397
  %v399 = load double, double* %v15, align 8
  %v400 = load double, double* %v29, align 8
  %v401 = fmul double %v399, %v400
  %v402 = fadd double %v398, %v401
  %v403 = load double, double* %v13, align 8
  %v404 = load double, double* %v28, align 8
  %v405 = fmul double %v403, %v404
  %v406 = fadd double %v402, %v405
  %v407 = load double, double* %v14, align 8
  %v408 = load double, double* %v26, align 8
  %v409 = fmul double %v407, %v408
  %v410 = fsub double %v406, %v409
  %v411 = load double, double* %v15, align 8
  %v412 = load double, double* %v25, align 8
  %v413 = fmul double %v411, %v412
  %v414 = fsub double %v410, %v413
  store double %v414, double* %v33, align 8
  %v415 = load double, double* %v33, align 8
  %v416 = load double, double* %v32, align 8
  %v417 = fmul double %v415, %v416
  %v418 = fptrunc double %v417 to float
  store float %v418, float* %v9, align 4
  %v419 = load double, double* %v13, align 8
  %v420 = load double, double* %v20, align 8
  %v421 = fmul double %v419, %v420
  %v422 = load double, double* %v16, align 8
  %v423 = load double, double* %v23, align 8
  %v424 = fmul double %v422, %v423
  %v425 = fsub double %v421, %v424
  %v426 = load double, double* %v17, align 8
  %v427 = load double, double* %v24, align 8
  %v428 = fmul double %v426, %v427
  %v429 = fsub double %v425, %v428
  %v430 = load double, double* %v18, align 8
  %v431 = load double, double* %v27, align 8
  %v432 = fmul double %v430, %v431
  %v433 = fadd double %v429, %v432
  %v434 = load double, double* %v19, align 8
  %v435 = load double, double* %v28, align 8
  %v436 = fmul double %v434, %v435
  %v437 = fadd double %v433, %v436
  store double %v437, double* %v33, align 8
  %v438 = load double, double* %v33, align 8
  %v439 = load double, double* %v32, align 8
  %v440 = fmul double %v438, %v439
  %v441 = fptrunc double %v440 to float
  store float %v441, float* %v10, align 4
  %v442 = load double, double* %v18, align 8
  %v443 = fsub double -0.000000e+00, %v442
  %v444 = load double, double* %v29, align 8
  %v445 = fmul double %v443, %v444
  %v446 = load double, double* %v19, align 8
  %v447 = load double, double* %v30, align 8
  %v448 = fmul double %v446, %v447
  %v449 = fsub double %v445, %v448
  %v450 = load double, double* %v14, align 8
  %v451 = load double, double* %v23, align 8
  %v452 = fmul double %v450, %v451
  %v453 = fadd double %v449, %v452
  %v454 = load double, double* %v15, align 8
  %v455 = load double, double* %v24, align 8
  %v456 = fmul double %v454, %v455
  %v457 = fadd double %v453, %v456
  %v458 = load double, double* %v13, align 8
  %v459 = load double, double* %v21, align 8
  %v460 = fmul double %v458, %v459
  %v461 = fsub double %v457, %v460
  store double %v461, double* %v33, align 8
  %v462 = load double, double* %v33, align 8
  %v463 = load double, double* %v32, align 8
  %v464 = fmul double %v462, %v463
  %v465 = fptrunc double %v464 to float
  store float %v465, float* %v11, align 4
  %v466 = load double, double* %v18, align 8
  %v467 = fsub double -0.000000e+00, %v466
  %v468 = load double, double* %v30, align 8
  %v469 = fmul double %v467, %v468
  %v470 = load double, double* %v19, align 8
  %v471 = load double, double* %v29, align 8
  %v472 = fmul double %v470, %v471
  %v473 = fadd double %v469, %v472
  %v474 = load double, double* %v14, align 8
  %v475 = load double, double* %v24, align 8
  %v476 = fmul double %v474, %v475
  %v477 = fadd double %v473, %v476
  %v478 = load double, double* %v15, align 8
  %v479 = load double, double* %v23, align 8
  %v480 = fmul double %v478, %v479
  %v481 = fsub double %v477, %v480
  %v482 = load double, double* %v13, align 8
  %v483 = load double, double* %v22, align 8
  %v484 = fmul double %v482, %v483
  %v485 = fsub double %v481, %v484
  store double %v485, double* %v33, align 8
  %v486 = load double, double* %v33, align 8
  %v487 = load double, double* %v32, align 8
  %v488 = fmul double %v486, %v487
  %v489 = fptrunc double %v488 to float
  store float %v489, float* %v12, align 4
  %v490 = load float, float* %v3, align 4
  %v491 = load %s.12*, %s.12** %v2, align 4
  %v492 = getelementptr inbounds %s.12, %s.12* %v491, i32 0
  %v493 = getelementptr inbounds %s.12, %s.12* %v492, i32 0, i32 0
  store float %v490, float* %v493, align 4
  %v494 = load %s.12*, %s.12** %v2, align 4
  %v495 = getelementptr inbounds %s.12, %s.12* %v494, i32 0
  %v496 = getelementptr inbounds %s.12, %s.12* %v495, i32 0, i32 1
  store float 0.000000e+00, float* %v496, align 4
  %v497 = load float, float* %v4, align 4
  %v498 = load %s.12*, %s.12** %v2, align 4
  %v499 = getelementptr inbounds %s.12, %s.12* %v498, i32 1
  %v500 = getelementptr inbounds %s.12, %s.12* %v499, i32 0, i32 0
  store float %v497, float* %v500, align 4
  %v501 = load float, float* %v5, align 4
  %v502 = load %s.12*, %s.12** %v2, align 4
  %v503 = getelementptr inbounds %s.12, %s.12* %v502, i32 1
  %v504 = getelementptr inbounds %s.12, %s.12* %v503, i32 0, i32 1
  store float %v501, float* %v504, align 4
  %v505 = load float, float* %v6, align 4
  %v506 = load %s.12*, %s.12** %v2, align 4
  %v507 = getelementptr inbounds %s.12, %s.12* %v506, i32 2
  %v508 = getelementptr inbounds %s.12, %s.12* %v507, i32 0, i32 0
  store float %v505, float* %v508, align 4
  %v509 = load float, float* %v7, align 4
  %v510 = load %s.12*, %s.12** %v2, align 4
  %v511 = getelementptr inbounds %s.12, %s.12* %v510, i32 2
  %v512 = getelementptr inbounds %s.12, %s.12* %v511, i32 0, i32 1
  store float %v509, float* %v512, align 4
  %v513 = load float, float* %v8, align 4
  %v514 = load %s.12*, %s.12** %v2, align 4
  %v515 = getelementptr inbounds %s.12, %s.12* %v514, i32 3
  %v516 = getelementptr inbounds %s.12, %s.12* %v515, i32 0, i32 0
  store float %v513, float* %v516, align 4
  %v517 = load float, float* %v9, align 4
  %v518 = load %s.12*, %s.12** %v2, align 4
  %v519 = getelementptr inbounds %s.12, %s.12* %v518, i32 3
  %v520 = getelementptr inbounds %s.12, %s.12* %v519, i32 0, i32 1
  store float %v517, float* %v520, align 4
  %v521 = load float, float* %v4, align 4
  %v522 = load %s.12*, %s.12** %v2, align 4
  %v523 = getelementptr inbounds %s.12, %s.12* %v522, i32 4
  %v524 = getelementptr inbounds %s.12, %s.12* %v523, i32 0, i32 0
  store float %v521, float* %v524, align 4
  %v525 = load float, float* %v5, align 4
  %v526 = fsub float -0.000000e+00, %v525
  %v527 = load %s.12*, %s.12** %v2, align 4
  %v528 = getelementptr inbounds %s.12, %s.12* %v527, i32 4
  %v529 = getelementptr inbounds %s.12, %s.12* %v528, i32 0, i32 1
  store float %v526, float* %v529, align 4
  %v530 = load float, float* %v10, align 4
  %v531 = load %s.12*, %s.12** %v2, align 4
  %v532 = getelementptr inbounds %s.12, %s.12* %v531, i32 5
  %v533 = getelementptr inbounds %s.12, %s.12* %v532, i32 0, i32 0
  store float %v530, float* %v533, align 4
  %v534 = load %s.12*, %s.12** %v2, align 4
  %v535 = getelementptr inbounds %s.12, %s.12* %v534, i32 5
  %v536 = getelementptr inbounds %s.12, %s.12* %v535, i32 0, i32 1
  store float 0.000000e+00, float* %v536, align 4
  %v537 = load float, float* %v11, align 4
  %v538 = load %s.12*, %s.12** %v2, align 4
  %v539 = getelementptr inbounds %s.12, %s.12* %v538, i32 6
  %v540 = getelementptr inbounds %s.12, %s.12* %v539, i32 0, i32 0
  store float %v537, float* %v540, align 4
  %v541 = load float, float* %v12, align 4
  %v542 = load %s.12*, %s.12** %v2, align 4
  %v543 = getelementptr inbounds %s.12, %s.12* %v542, i32 6
  %v544 = getelementptr inbounds %s.12, %s.12* %v543, i32 0, i32 1
  store float %v541, float* %v544, align 4
  %v545 = load float, float* %v6, align 4
  %v546 = load %s.12*, %s.12** %v2, align 4
  %v547 = getelementptr inbounds %s.12, %s.12* %v546, i32 7
  %v548 = getelementptr inbounds %s.12, %s.12* %v547, i32 0, i32 0
  store float %v545, float* %v548, align 4
  %v549 = load float, float* %v7, align 4
  %v550 = load %s.12*, %s.12** %v2, align 4
  %v551 = getelementptr inbounds %s.12, %s.12* %v550, i32 7
  %v552 = getelementptr inbounds %s.12, %s.12* %v551, i32 0, i32 1
  store float %v549, float* %v552, align 4
  %v553 = load float, float* %v6, align 4
  %v554 = load %s.12*, %s.12** %v2, align 4
  %v555 = getelementptr inbounds %s.12, %s.12* %v554, i32 8
  %v556 = getelementptr inbounds %s.12, %s.12* %v555, i32 0, i32 0
  store float %v553, float* %v556, align 4
  %v557 = load float, float* %v7, align 4
  %v558 = fsub float -0.000000e+00, %v557
  %v559 = load %s.12*, %s.12** %v2, align 4
  %v560 = getelementptr inbounds %s.12, %s.12* %v559, i32 8
  %v561 = getelementptr inbounds %s.12, %s.12* %v560, i32 0, i32 1
  store float %v558, float* %v561, align 4
  %v562 = load float, float* %v11, align 4
  %v563 = load %s.12*, %s.12** %v2, align 4
  %v564 = getelementptr inbounds %s.12, %s.12* %v563, i32 9
  %v565 = getelementptr inbounds %s.12, %s.12* %v564, i32 0, i32 0
  store float %v562, float* %v565, align 4
  %v566 = load float, float* %v12, align 4
  %v567 = fsub float -0.000000e+00, %v566
  %v568 = load %s.12*, %s.12** %v2, align 4
  %v569 = getelementptr inbounds %s.12, %s.12* %v568, i32 9
  %v570 = getelementptr inbounds %s.12, %s.12* %v569, i32 0, i32 1
  store float %v567, float* %v570, align 4
  %v571 = load float, float* %v10, align 4
  %v572 = load %s.12*, %s.12** %v2, align 4
  %v573 = getelementptr inbounds %s.12, %s.12* %v572, i32 10
  %v574 = getelementptr inbounds %s.12, %s.12* %v573, i32 0, i32 0
  store float %v571, float* %v574, align 4
  %v575 = load %s.12*, %s.12** %v2, align 4
  %v576 = getelementptr inbounds %s.12, %s.12* %v575, i32 10
  %v577 = getelementptr inbounds %s.12, %s.12* %v576, i32 0, i32 1
  store float 0.000000e+00, float* %v577, align 4
  %v578 = load float, float* %v4, align 4
  %v579 = load %s.12*, %s.12** %v2, align 4
  %v580 = getelementptr inbounds %s.12, %s.12* %v579, i32 11
  %v581 = getelementptr inbounds %s.12, %s.12* %v580, i32 0, i32 0
  store float %v578, float* %v581, align 4
  %v582 = load float, float* %v5, align 4
  %v583 = load %s.12*, %s.12** %v2, align 4
  %v584 = getelementptr inbounds %s.12, %s.12* %v583, i32 11
  %v585 = getelementptr inbounds %s.12, %s.12* %v584, i32 0, i32 1
  store float %v582, float* %v585, align 4
  %v586 = load float, float* %v8, align 4
  %v587 = load %s.12*, %s.12** %v2, align 4
  %v588 = getelementptr inbounds %s.12, %s.12* %v587, i32 12
  %v589 = getelementptr inbounds %s.12, %s.12* %v588, i32 0, i32 0
  store float %v586, float* %v589, align 4
  %v590 = load float, float* %v9, align 4
  %v591 = fsub float -0.000000e+00, %v590
  %v592 = load %s.12*, %s.12** %v2, align 4
  %v593 = getelementptr inbounds %s.12, %s.12* %v592, i32 12
  %v594 = getelementptr inbounds %s.12, %s.12* %v593, i32 0, i32 1
  store float %v591, float* %v594, align 4
  %v595 = load float, float* %v6, align 4
  %v596 = load %s.12*, %s.12** %v2, align 4
  %v597 = getelementptr inbounds %s.12, %s.12* %v596, i32 13
  %v598 = getelementptr inbounds %s.12, %s.12* %v597, i32 0, i32 0
  store float %v595, float* %v598, align 4
  %v599 = load float, float* %v7, align 4
  %v600 = fsub float -0.000000e+00, %v599
  %v601 = load %s.12*, %s.12** %v2, align 4
  %v602 = getelementptr inbounds %s.12, %s.12* %v601, i32 13
  %v603 = getelementptr inbounds %s.12, %s.12* %v602, i32 0, i32 1
  store float %v600, float* %v603, align 4
  %v604 = load float, float* %v4, align 4
  %v605 = load %s.12*, %s.12** %v2, align 4
  %v606 = getelementptr inbounds %s.12, %s.12* %v605, i32 14
  %v607 = getelementptr inbounds %s.12, %s.12* %v606, i32 0, i32 0
  store float %v604, float* %v607, align 4
  %v608 = load float, float* %v5, align 4
  %v609 = fsub float -0.000000e+00, %v608
  %v610 = load %s.12*, %s.12** %v2, align 4
  %v611 = getelementptr inbounds %s.12, %s.12* %v610, i32 14
  %v612 = getelementptr inbounds %s.12, %s.12* %v611, i32 0, i32 1
  store float %v609, float* %v612, align 4
  %v613 = load float, float* %v3, align 4
  %v614 = load %s.12*, %s.12** %v2, align 4
  %v615 = getelementptr inbounds %s.12, %s.12* %v614, i32 15
  %v616 = getelementptr inbounds %s.12, %s.12* %v615, i32 0, i32 0
  store float %v613, float* %v616, align 4
  %v617 = load %s.12*, %s.12** %v2, align 4
  %v618 = getelementptr inbounds %s.12, %s.12* %v617, i32 15
  %v619 = getelementptr inbounds %s.12, %s.12* %v618, i32 0, i32 1
  store float 0.000000e+00, float* %v619, align 4
  br label %b3

b3:                                               ; preds = %b2, %b1
  ret void
}

; Function Attrs: nounwind readnone
declare double @f1(double) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
