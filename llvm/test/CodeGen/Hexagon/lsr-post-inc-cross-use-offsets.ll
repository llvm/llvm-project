; RUN: llc -O3 -march=hexagon < %s | FileCheck %s

; CHECK: loop0(.[[BLOCK:LBB0_[0-9]+]]
; CHECK: .[[BLOCK]]:
; CHECK: = vmemu({{r[0-9]+}}++#1)
; CHECK: = vmemu({{r[0-9]+}}++#1)
; CHECK: = vmemu({{r[0-9]+}}++#1)
; CHECK: = vmemu({{r[0-9]+}}++#1)
; CHECK: endloop0

target triple = "hexagon-unknown--elf"

%0 = type { ptr, i32, i32, i32, i32, ptr, ptr, ptr }
%1 = type { %2 }
%2 = type { i64 }
%3 = type { ptr, i32, i32, i32, i32, i32, i32, ptr, i32, ptr }
%4 = type { i64, ptr, [4 x i32], [4 x i32], [4 x i32], i32, i8, i8, [6 x i8] }

@g0 = private unnamed_addr constant [5 x i8] c"Load\00", align 1
@g1 = private unnamed_addr constant [6 x i8] c"Store\00", align 1
@g2 = private unnamed_addr constant [18 x i8] c"Begin realization\00", align 1
@g3 = private unnamed_addr constant [16 x i8] c"End realization\00", align 1
@g4 = private unnamed_addr constant [8 x i8] c"Produce\00", align 1
@g5 = private unnamed_addr constant [7 x i8] c"Update\00", align 1
@g6 = private unnamed_addr constant [8 x i8] c"Consume\00", align 1
@g7 = private unnamed_addr constant [12 x i8] c"End consume\00", align 1
@g8 = private constant [6 x i8] c"input\00", align 32
@g9 = private constant [10 x i8] c"dilate3x3\00", align 32
@g10 = private constant [2 x %0] [%0 { ptr @g8, i32 1, i32 2, i32 1, i32 8, ptr null, ptr null, ptr null }, %0 { ptr @g9, i32 2, i32 2, i32 1, i32 8, ptr null, ptr null, ptr null }]
@g11 = private constant [64 x i8] c"...............................................................\00", align 32

; Function Attrs: nounwind
declare ptr @f0(ptr, i32) #0

; Function Attrs: nounwind
declare void @f1(ptr, ptr) #0

; Function Attrs: nounwind
declare void @f2(ptr, ptr) #0

; Function Attrs: nounwind
declare i32 @f3(ptr, ptr) #0

; Function Attrs: nounwind
declare void @f4() #0

; Function Attrs: nounwind
declare void @f5() #0

; Function Attrs: nounwind
define i32 @f6(ptr noalias nocapture readonly %a0, ptr noalias nocapture readonly %a1) #0 {
b0:
  %v0 = getelementptr inbounds %4, ptr %a0, i32 0, i32 1
  %v1 = load ptr, ptr %v0, align 4
  %v2 = getelementptr inbounds %4, ptr %a0, i32 0, i32 3, i32 1
  %v3 = load i32, ptr %v2, align 4
  %v4 = getelementptr inbounds %4, ptr %a0, i32 0, i32 4, i32 0
  %v5 = load i32, ptr %v4, align 4
  %v6 = getelementptr inbounds %4, ptr %a0, i32 0, i32 4, i32 1
  %v7 = load i32, ptr %v6, align 4
  %v8 = getelementptr inbounds %4, ptr %a1, i32 0, i32 1
  %v9 = load ptr, ptr %v8, align 4
  %v10 = getelementptr inbounds %4, ptr %a1, i32 0, i32 2, i32 0
  %v11 = load i32, ptr %v10, align 4
  %v12 = getelementptr inbounds %4, ptr %a1, i32 0, i32 3, i32 1
  %v13 = load i32, ptr %v12, align 4
  %v14 = getelementptr inbounds %4, ptr %a1, i32 0, i32 4, i32 0
  %v15 = load i32, ptr %v14, align 4
  %v16 = getelementptr inbounds %4, ptr %a1, i32 0, i32 4, i32 1
  %v17 = load i32, ptr %v16, align 4
  %v18 = getelementptr inbounds %4, ptr %a1, i32 0, i32 2, i32 1
  %v19 = load i32, ptr %v18, align 4
  %v20 = add nsw i32 %v19, %v17
  %v21 = icmp sgt i32 %v19, 0
  br i1 %v21, label %b1, label %b11, !prof !3

b1:                                               ; preds = %b0
  %v22 = ashr i32 %v11, 7
  %v23 = icmp slt i32 %v22, 0
  %v24 = select i1 %v23, i32 0, i32 %v22
  %v25 = icmp sgt i32 %v24, 0
  br i1 %v25, label %b5, label %b7, !prof !3

b2:                                               ; preds = %b5, %b2
  %v26 = phi i32 [ %v90, %b2 ], [ 0, %b5 ]
  %v27 = mul nsw i32 %v7, %v3
  %v28 = add nsw i32 %v27, %v5
  %v29 = shl nsw i32 %v26, 7
  %v30 = add nsw i32 %v29, %v15
  %v31 = add nsw i32 %v150, -1
  %v32 = mul nsw i32 %v31, %v3
  %v33 = mul nsw i32 %v150, %v3
  %v34 = add nsw i32 %v150, 1
  %v35 = mul nsw i32 %v34, %v3
  %v36 = sub i32 %v32, %v28
  %v37 = add i32 %v36, %v30
  %v38 = add nsw i32 %v37, -1
  %v39 = getelementptr inbounds i8, ptr %v1, i32 %v38
  %v41 = load <32 x i32>, ptr %v39, align 1, !tbaa !4
  %v42 = getelementptr inbounds i8, ptr %v1, i32 %v37
  %v44 = load <32 x i32>, ptr %v42, align 1, !tbaa !4
  %v45 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v41, <32 x i32> %v44)
  %v46 = add nsw i32 %v37, 1
  %v47 = getelementptr inbounds i8, ptr %v1, i32 %v46
  %v49 = load <32 x i32>, ptr %v47, align 1, !tbaa !4
  %v50 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v45, <32 x i32> %v49)
  %v51 = sub i32 %v33, %v28
  %v52 = add i32 %v51, %v30
  %v53 = add nsw i32 %v52, -1
  %v54 = getelementptr inbounds i8, ptr %v1, i32 %v53
  %v56 = load <32 x i32>, ptr %v54, align 1, !tbaa !4
  %v57 = getelementptr inbounds i8, ptr %v1, i32 %v52
  %v59 = load <32 x i32>, ptr %v57, align 1, !tbaa !4
  %v60 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v56, <32 x i32> %v59)
  %v61 = add nsw i32 %v52, 1
  %v62 = getelementptr inbounds i8, ptr %v1, i32 %v61
  %v64 = load <32 x i32>, ptr %v62, align 1, !tbaa !4
  %v65 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v60, <32 x i32> %v64)
  %v66 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v50, <32 x i32> %v65)
  %v67 = sub i32 %v35, %v28
  %v68 = add i32 %v67, %v30
  %v69 = add nsw i32 %v68, -1
  %v70 = getelementptr inbounds i8, ptr %v1, i32 %v69
  %v72 = load <32 x i32>, ptr %v70, align 1, !tbaa !4
  %v73 = getelementptr inbounds i8, ptr %v1, i32 %v68
  %v75 = load <32 x i32>, ptr %v73, align 1, !tbaa !4
  %v76 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v72, <32 x i32> %v75)
  %v77 = add nsw i32 %v68, 1
  %v78 = getelementptr inbounds i8, ptr %v1, i32 %v77
  %v80 = load <32 x i32>, ptr %v78, align 1, !tbaa !4
  %v81 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v76, <32 x i32> %v80)
  %v82 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v66, <32 x i32> %v81)
  %v83 = mul nsw i32 %v150, %v13
  %v84 = mul nsw i32 %v17, %v13
  %v85 = add i32 %v84, %v15
  %v86 = sub i32 %v83, %v85
  %v87 = add i32 %v86, %v30
  %v88 = getelementptr inbounds i8, ptr %v9, i32 %v87
  store <32 x i32> %v82, ptr %v88, align 1, !tbaa !7
  %v90 = add nuw nsw i32 %v26, 1
  %v91 = icmp eq i32 %v90, %v24
  br i1 %v91, label %b6, label %b2

b3:                                               ; preds = %b6, %b3
  %v92 = phi i32 [ %v147, %b3 ], [ %v24, %b6 ]
  %v93 = add nsw i32 %v15, %v11
  %v94 = sub i32 %v93, %v28
  %v95 = add i32 %v94, %v32
  %v96 = add nsw i32 %v95, -129
  %v97 = getelementptr inbounds i8, ptr %v1, i32 %v96
  %v99 = load <32 x i32>, ptr %v97, align 1, !tbaa !4
  %v100 = add nsw i32 %v95, -128
  %v101 = getelementptr inbounds i8, ptr %v1, i32 %v100
  %v103 = load <32 x i32>, ptr %v101, align 1, !tbaa !4
  %v104 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v99, <32 x i32> %v103)
  %v105 = add nsw i32 %v95, -127
  %v106 = getelementptr inbounds i8, ptr %v1, i32 %v105
  %v108 = load <32 x i32>, ptr %v106, align 1, !tbaa !4
  %v109 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v104, <32 x i32> %v108)
  %v110 = add i32 %v94, %v33
  %v111 = add nsw i32 %v110, -129
  %v112 = getelementptr inbounds i8, ptr %v1, i32 %v111
  %v114 = load <32 x i32>, ptr %v112, align 1, !tbaa !4
  %v115 = add nsw i32 %v110, -128
  %v116 = getelementptr inbounds i8, ptr %v1, i32 %v115
  %v118 = load <32 x i32>, ptr %v116, align 1, !tbaa !4
  %v119 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v114, <32 x i32> %v118)
  %v120 = add nsw i32 %v110, -127
  %v121 = getelementptr inbounds i8, ptr %v1, i32 %v120
  %v123 = load <32 x i32>, ptr %v121, align 1, !tbaa !4
  %v124 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v119, <32 x i32> %v123)
  %v125 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v109, <32 x i32> %v124)
  %v126 = add i32 %v94, %v35
  %v127 = add nsw i32 %v126, -129
  %v128 = getelementptr inbounds i8, ptr %v1, i32 %v127
  %v130 = load <32 x i32>, ptr %v128, align 1, !tbaa !4
  %v131 = add nsw i32 %v126, -128
  %v132 = getelementptr inbounds i8, ptr %v1, i32 %v131
  %v134 = load <32 x i32>, ptr %v132, align 1, !tbaa !4
  %v135 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v130, <32 x i32> %v134)
  %v136 = add nsw i32 %v126, -127
  %v137 = getelementptr inbounds i8, ptr %v1, i32 %v136
  %v139 = load <32 x i32>, ptr %v137, align 1, !tbaa !4
  %v140 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v135, <32 x i32> %v139)
  %v141 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v125, <32 x i32> %v140)
  %v142 = add i32 %v11, -128
  %v143 = sub i32 %v142, %v84
  %v144 = add i32 %v143, %v83
  %v145 = getelementptr inbounds i8, ptr %v9, i32 %v144
  store <32 x i32> %v141, ptr %v145, align 1, !tbaa !7
  %v147 = add nuw nsw i32 %v92, 1
  %v148 = icmp eq i32 %v147, %v152
  br i1 %v148, label %b4, label %b3

b4:                                               ; preds = %b6, %b3
  %v149 = icmp eq i32 %v34, %v20
  br i1 %v149, label %b11, label %b5

b5:                                               ; preds = %b4, %b1
  %v150 = phi i32 [ %v34, %b4 ], [ %v17, %b1 ]
  br label %b2

b6:                                               ; preds = %b2
  %v151 = add nsw i32 %v11, 127
  %v152 = ashr i32 %v151, 7
  %v153 = icmp slt i32 %v24, %v152
  br i1 %v153, label %b3, label %b4, !prof !3

b7:                                               ; preds = %b1
  %v154 = add nsw i32 %v11, 127
  %v155 = ashr i32 %v154, 7
  %v156 = icmp slt i32 %v24, %v155
  br i1 %v156, label %b9, label %b11, !prof !3

b8:                                               ; preds = %b9, %b8
  %v157 = phi i32 [ %v221, %b8 ], [ %v24, %b9 ]
  %v158 = mul nsw i32 %v7, %v3
  %v159 = add nsw i32 %v158, %v5
  %v160 = add nsw i32 %v15, %v11
  %v161 = add nsw i32 %v223, -1
  %v162 = mul nsw i32 %v161, %v3
  %v163 = mul nsw i32 %v223, %v3
  %v164 = add nsw i32 %v223, 1
  %v165 = mul nsw i32 %v164, %v3
  %v166 = sub i32 %v160, %v159
  %v167 = add i32 %v166, %v162
  %v168 = add nsw i32 %v167, -129
  %v169 = getelementptr inbounds i8, ptr %v1, i32 %v168
  %v171 = load <32 x i32>, ptr %v169, align 1, !tbaa !4
  %v172 = add nsw i32 %v167, -128
  %v173 = getelementptr inbounds i8, ptr %v1, i32 %v172
  %v175 = load <32 x i32>, ptr %v173, align 1, !tbaa !4
  %v176 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v171, <32 x i32> %v175)
  %v177 = add nsw i32 %v167, -127
  %v178 = getelementptr inbounds i8, ptr %v1, i32 %v177
  %v180 = load <32 x i32>, ptr %v178, align 1, !tbaa !4
  %v181 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v176, <32 x i32> %v180)
  %v182 = add i32 %v166, %v163
  %v183 = add nsw i32 %v182, -129
  %v184 = getelementptr inbounds i8, ptr %v1, i32 %v183
  %v186 = load <32 x i32>, ptr %v184, align 1, !tbaa !4
  %v187 = add nsw i32 %v182, -128
  %v188 = getelementptr inbounds i8, ptr %v1, i32 %v187
  %v190 = load <32 x i32>, ptr %v188, align 1, !tbaa !4
  %v191 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v186, <32 x i32> %v190)
  %v192 = add nsw i32 %v182, -127
  %v193 = getelementptr inbounds i8, ptr %v1, i32 %v192
  %v195 = load <32 x i32>, ptr %v193, align 1, !tbaa !4
  %v196 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v191, <32 x i32> %v195)
  %v197 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v181, <32 x i32> %v196)
  %v198 = add i32 %v166, %v165
  %v199 = add nsw i32 %v198, -129
  %v200 = getelementptr inbounds i8, ptr %v1, i32 %v199
  %v202 = load <32 x i32>, ptr %v200, align 1, !tbaa !4
  %v203 = add nsw i32 %v198, -128
  %v204 = getelementptr inbounds i8, ptr %v1, i32 %v203
  %v206 = load <32 x i32>, ptr %v204, align 1, !tbaa !4
  %v207 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v202, <32 x i32> %v206)
  %v208 = add nsw i32 %v198, -127
  %v209 = getelementptr inbounds i8, ptr %v1, i32 %v208
  %v211 = load <32 x i32>, ptr %v209, align 1, !tbaa !4
  %v212 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v207, <32 x i32> %v211)
  %v213 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %v197, <32 x i32> %v212)
  %v214 = mul nsw i32 %v223, %v13
  %v215 = mul nsw i32 %v17, %v13
  %v216 = add i32 %v11, -128
  %v217 = sub i32 %v216, %v215
  %v218 = add i32 %v217, %v214
  %v219 = getelementptr inbounds i8, ptr %v9, i32 %v218
  store <32 x i32> %v213, ptr %v219, align 1, !tbaa !7
  %v221 = add nuw nsw i32 %v157, 1
  %v222 = icmp eq i32 %v221, %v155
  br i1 %v222, label %b10, label %b8

b9:                                               ; preds = %b10, %b7
  %v223 = phi i32 [ %v164, %b10 ], [ %v17, %b7 ]
  br label %b8

b10:                                              ; preds = %b8
  %v224 = icmp eq i32 %v164, %v20
  br i1 %v224, label %b11, label %b9

b11:                                              ; preds = %b10, %b7, %b4, %b0
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind
define i32 @f7(ptr noalias nocapture readonly %a0, ptr noalias nocapture readonly %a1) #0 {
b0:
  %v0 = tail call i32 @f6(ptr %a0, ptr %a1) #0
  ret i32 0
}

; Function Attrs: nounwind
define i32 @f8(ptr nocapture readonly %a0) #0 {
b0:
  %v1 = load ptr, ptr %a0, align 4
  %v2 = getelementptr ptr, ptr %a0, i32 1
  %v4 = load ptr, ptr %v2, align 4
  %v5 = tail call i32 @f7(ptr %v1, ptr %v4)
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"halide_use_soft_float_abi", i32 0}
!1 = !{i32 2, !"halide_mcpu", !"hexagonv60"}
!2 = !{i32 2, !"halide_mattrs", !"+hvxv60,+hvx-length64b"}
!3 = !{!"branch_weights", i32 1073741824, i32 0}
!4 = !{!5, !5, i64 0}
!5 = !{!"input", !6}
!6 = !{!"Halide buffer"}
!7 = !{!8, !8, i64 0}
!8 = !{!"dilate3x3", !6}
