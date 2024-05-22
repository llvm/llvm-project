; RUN: llc < %s -mtriple=x86_64-apple-darwin8 -misched-print-dags -o - 2>&1 > /dev/null | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

@sc = common global i8 0
@uc = common global i8 0
@ss = common global i16 0
@us = common global i16 0
@si = common global i32 0
@ui = common global i32 0
@sl = common global i64 0
@ul = common global i64 0
@sll = common global i64 0
@ull = common global i64 0

; Regression Test for PR92368.
;
; CHECK: SU(75):   CMP8rr %49:gr8, %48:gr8, implicit-def $eflags
; CHECK:   Predecessors:
; CHECK-NEXT:    SU(73): Data Latency=0 Reg=%49
; CHECK-NEXT:    SU(74): Out  Latency=0
; CHECK-NEXT:    SU(72): Out  Latency=0
; CHECK-NEXT:    SU(70): Data Latency=4 Reg=%48
define void @misched_bug() nounwind {
entry:
  %v0 = load i8, ptr @sc, align 1
  %v1 = zext i8 %v0 to i32
  %v2 = load i8, ptr @uc, align 1
  %v3 = zext i8 %v2 to i32
  %v4 = trunc i32 %v3 to i8
  %v5 = trunc i32 %v1 to i8
  %pair6 = cmpxchg ptr @sc, i8 %v4, i8 %v5 monotonic monotonic
  %v6 = extractvalue { i8, i1 } %pair6, 0
  store i8 %v6, ptr @sc, align 1
  %v7 = load i8, ptr @sc, align 1
  %v8 = zext i8 %v7 to i32
  %v9 = load i8, ptr @uc, align 1
  %v10 = zext i8 %v9 to i32
  %v11 = trunc i32 %v10 to i8
  %v12 = trunc i32 %v8 to i8
  %pair13 = cmpxchg ptr @uc, i8 %v11, i8 %v12 monotonic monotonic
  %v13 = extractvalue { i8, i1 } %pair13, 0
  store i8 %v13, ptr @uc, align 1
  %v14 = load i8, ptr @sc, align 1
  %v15 = sext i8 %v14 to i16
  %v16 = zext i16 %v15 to i32
  %v17 = load i8, ptr @uc, align 1
  %v18 = zext i8 %v17 to i32
  %v20 = trunc i32 %v18 to i16
  %v21 = trunc i32 %v16 to i16
  %pair22 = cmpxchg ptr @ss, i16 %v20, i16 %v21 monotonic monotonic
  %v22 = extractvalue { i16, i1 } %pair22, 0
  store i16 %v22, ptr @ss, align 2
  %v23 = load i8, ptr @sc, align 1
  %v24 = sext i8 %v23 to i16
  %v25 = zext i16 %v24 to i32
  %v26 = load i8, ptr @uc, align 1
  %v27 = zext i8 %v26 to i32
  %v29 = trunc i32 %v27 to i16
  %v30 = trunc i32 %v25 to i16
  %pair31 = cmpxchg ptr @us, i16 %v29, i16 %v30 monotonic monotonic
  %v31 = extractvalue { i16, i1 } %pair31, 0
  store i16 %v31, ptr @us, align 2
  %v32 = load i8, ptr @sc, align 1
  %v33 = sext i8 %v32 to i32
  %v34 = load i8, ptr @uc, align 1
  %v35 = zext i8 %v34 to i32
  %pair37 = cmpxchg ptr @si, i32 %v35, i32 %v33 monotonic monotonic
  %v37 = extractvalue { i32, i1 } %pair37, 0
  store i32 %v37, ptr @si, align 4
  %v38 = load i8, ptr @sc, align 1
  %v39 = sext i8 %v38 to i32
  %v40 = load i8, ptr @uc, align 1
  %v41 = zext i8 %v40 to i32
  %pair43 = cmpxchg ptr @ui, i32 %v41, i32 %v39 monotonic monotonic
  %v43 = extractvalue { i32, i1 } %pair43, 0
  store i32 %v43, ptr @ui, align 4
  %v44 = load i8, ptr @sc, align 1
  %v45 = sext i8 %v44 to i64
  %v46 = load i8, ptr @uc, align 1
  %v47 = zext i8 %v46 to i64
  %pair49 = cmpxchg ptr @sl, i64 %v47, i64 %v45 monotonic monotonic
  %v49 = extractvalue { i64, i1 } %pair49, 0
  store i64 %v49, ptr @sl, align 8
  %v50 = load i8, ptr @sc, align 1
  %v51 = sext i8 %v50 to i64
  %v52 = load i8, ptr @uc, align 1
  %v53 = zext i8 %v52 to i64
  %pair55 = cmpxchg ptr @ul, i64 %v53, i64 %v51 monotonic monotonic
  %v55 = extractvalue { i64, i1 } %pair55, 0
  store i64 %v55, ptr @ul, align 8
  %v56 = load i8, ptr @sc, align 1
  %v57 = sext i8 %v56 to i64
  %v58 = load i8, ptr @uc, align 1
  %v59 = zext i8 %v58 to i64
  %pair61 = cmpxchg ptr @sll, i64 %v59, i64 %v57 monotonic monotonic
  %v61 = extractvalue { i64, i1 } %pair61, 0
  store i64 %v61, ptr @sll, align 8
  %v62 = load i8, ptr @sc, align 1
  %v63 = sext i8 %v62 to i64
  %v64 = load i8, ptr @uc, align 1
  %v65 = zext i8 %v64 to i64
  %pair67 = cmpxchg ptr @ull, i64 %v65, i64 %v63 monotonic monotonic
  %v67 = extractvalue { i64, i1 } %pair67, 0
  store i64 %v67, ptr @ull, align 8
  %v68 = load i8, ptr @sc, align 1
  %v69 = zext i8 %v68 to i32
  %v70 = load i8, ptr @uc, align 1
  %v71 = zext i8 %v70 to i32
  %v72 = trunc i32 %v71 to i8
  %v73 = trunc i32 %v69 to i8
  %pair74 = cmpxchg ptr @sc, i8 %v72, i8 %v73 monotonic monotonic
  %v74 = extractvalue { i8, i1 } %pair74, 0
  %v75 = icmp eq i8 %v74, %v72
  %v76 = zext i1 %v75 to i8
  %v77 = zext i8 %v76 to i32
  store i32 %v77, ptr @ui, align 4
  %v78 = load i8, ptr @sc, align 1
  %v79 = zext i8 %v78 to i32
  %v80 = load i8, ptr @uc, align 1
  %v81 = zext i8 %v80 to i32
  %v82 = trunc i32 %v81 to i8
  %v83 = trunc i32 %v79 to i8
  %pair84 = cmpxchg ptr @uc, i8 %v82, i8 %v83 monotonic monotonic
  %v84 = extractvalue { i8, i1 } %pair84, 0
  %v85 = icmp eq i8 %v84, %v82
  %v86 = zext i1 %v85 to i8
  %v87 = zext i8 %v86 to i32
  store i32 %v87, ptr @ui, align 4
  %v88 = load i8, ptr @sc, align 1
  %v89 = sext i8 %v88 to i16
  %v90 = zext i16 %v89 to i32
  %v91 = load i8, ptr @uc, align 1
  %v92 = zext i8 %v91 to i32
  %v93 = trunc i32 %v92 to i8
  %v94 = trunc i32 %v90 to i8
  %pair95 = cmpxchg ptr @ss, i8 %v93, i8 %v94 monotonic monotonic
  %v95 = extractvalue { i8, i1 } %pair95, 0
  %v96 = icmp eq i8 %v95, %v93
  %v97 = zext i1 %v96 to i8
  %v98 = zext i8 %v97 to i32
  store i32 %v98, ptr @ui, align 4
  %v99 = load i8, ptr @sc, align 1
  %v100 = sext i8 %v99 to i16
  %v101 = zext i16 %v100 to i32
  %v102 = load i8, ptr @uc, align 1
  %v103 = zext i8 %v102 to i32
  %v104 = trunc i32 %v103 to i8
  %v105 = trunc i32 %v101 to i8
  %pair106 = cmpxchg ptr @us, i8 %v104, i8 %v105 monotonic monotonic
  %v106 = extractvalue { i8, i1 } %pair106, 0
  %v107 = icmp eq i8 %v106, %v104
  %v108 = zext i1 %v107 to i8
  %v109 = zext i8 %v108 to i32
  store i32 %v109, ptr @ui, align 4
  %v110 = load i8, ptr @sc, align 1
  %v111 = sext i8 %v110 to i32
  %v112 = load i8, ptr @uc, align 1
  %v113 = zext i8 %v112 to i32
  %v114 = trunc i32 %v113 to i8
  %v115 = trunc i32 %v111 to i8
  %pair116 = cmpxchg ptr @si, i8 %v114, i8 %v115 monotonic monotonic
  %v116 = extractvalue { i8, i1 } %pair116, 0
  %v117 = icmp eq i8 %v116, %v114
  %v118 = zext i1 %v117 to i8
  %v119 = zext i8 %v118 to i32
  store i32 %v119, ptr @ui, align 4
  %v120 = load i8, ptr @sc, align 1
  %v121 = sext i8 %v120 to i32
  %v122 = load i8, ptr @uc, align 1
  %v123 = zext i8 %v122 to i32
  %v124 = trunc i32 %v123 to i8
  %v125 = trunc i32 %v121 to i8
  %pair126 = cmpxchg ptr @ui, i8 %v124, i8 %v125 monotonic monotonic
  %v126 = extractvalue { i8, i1 } %pair126, 0
  %v127 = icmp eq i8 %v126, %v124
  %v128 = zext i1 %v127 to i8
  %v129 = zext i8 %v128 to i32
  store i32 %v129, ptr @ui, align 4
  %v130 = load i8, ptr @sc, align 1
  %v131 = sext i8 %v130 to i64
  %v132 = load i8, ptr @uc, align 1
  %v133 = zext i8 %v132 to i64
  %v134 = trunc i64 %v133 to i8
  %v135 = trunc i64 %v131 to i8
  %pair136 = cmpxchg ptr @sl, i8 %v134, i8 %v135 monotonic monotonic
  %v136 = extractvalue { i8, i1 } %pair136, 0
  %v137 = icmp eq i8 %v136, %v134
  %v138 = zext i1 %v137 to i8
  %v139 = zext i8 %v138 to i32
  store i32 %v139, ptr @ui, align 4
  %v140 = load i8, ptr @sc, align 1
  %v141 = sext i8 %v140 to i64
  %v142 = load i8, ptr @uc, align 1
  %v143 = zext i8 %v142 to i64
  %v144 = trunc i64 %v143 to i8
  %v145 = trunc i64 %v141 to i8
  %pair146 = cmpxchg ptr @ul, i8 %v144, i8 %v145 monotonic monotonic
  %v146 = extractvalue { i8, i1 } %pair146, 0
  %v147 = icmp eq i8 %v146, %v144
  %v148 = zext i1 %v147 to i8
  %v149 = zext i8 %v148 to i32
  store i32 %v149, ptr @ui, align 4
  %v150 = load i8, ptr @sc, align 1
  %v151 = sext i8 %v150 to i64
  %v152 = load i8, ptr @uc, align 1
  %v153 = zext i8 %v152 to i64
  %v154 = trunc i64 %v153 to i8
  %v155 = trunc i64 %v151 to i8
  %pair156 = cmpxchg ptr @sll, i8 %v154, i8 %v155 monotonic monotonic
  %v156 = extractvalue { i8, i1 } %pair156, 0
  %v157 = icmp eq i8 %v156, %v154
  %v158 = zext i1 %v157 to i8
  %v159 = zext i8 %v158 to i32
  store i32 %v159, ptr @ui, align 4
  %v160 = load i8, ptr @sc, align 1
  %v161 = sext i8 %v160 to i64
  %v162 = load i8, ptr @uc, align 1
  %v163 = zext i8 %v162 to i64
  %v164 = trunc i64 %v163 to i8
  %v165 = trunc i64 %v161 to i8
  %pair166 = cmpxchg ptr @ull, i8 %v164, i8 %v165 monotonic monotonic
  %v166 = extractvalue { i8, i1 } %pair166, 0
  %v167 = icmp eq i8 %v166, %v164
  %v168 = zext i1 %v167 to i8
  %v169 = zext i8 %v168 to i32
  store i32 %v169, ptr @ui, align 4
  br label %return

return:                                           ; preds = %ventry
  ret void
}

