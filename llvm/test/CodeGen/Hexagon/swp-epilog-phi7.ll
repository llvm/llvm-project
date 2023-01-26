; RUN: llc -opaque-pointers=0 -march=hexagon -O2 -enable-pipeliner -disable-block-placement=0 < %s  | FileCheck %s

; For the Phis generated in the epilog, test that we generate the correct
; names for the values coming from the prolog stages. The test belows
; checks that the value loaded in the first prolog block gets propagated
; through the first epilog to the use after the loop.

; CHECK: if ({{.*}}) jump
; CHECK: [[VREG:v([0-9]+)]]{{.*}} = {{.*}}vmem(r{{[0-9]+}}++#1)
; CHECK: if ({{.*}}) {{jump|jump:nt|jump:t}} [[EPLOG1:(.*)]]
; CHECK: if ({{.*}}) {{jump|jump:nt|jump:t}} [[EPLOG:(.*)]]
; CHECK: [[EPLOG]]:
; CHECK: [[VREG1:v([0-9]+)]] = [[VREG]]
; CHECK: [[VREG]] = v{{[0-9]+}}
; CHECK: [[EPLOG1]]:
; CHECK: = vlalign([[VREG]],[[VREG1]],#1)

; Function Attrs: nounwind
define void @f0(i8* noalias nocapture readonly %a0, i32 %a1, i32 %a2, i8* noalias nocapture readonly %a3, i32 %a4, i8* noalias nocapture %a5, i32 %a6) #0 {
b0:
  %v0 = sub i32 0, %a1
  %v1 = getelementptr inbounds i8, i8* %a0, i32 %v0
  %v2 = bitcast i8* %v1 to <16 x i32>*
  %v3 = bitcast i8* %a0 to <16 x i32>*
  %v4 = getelementptr inbounds i8, i8* %a0, i32 %a1
  %v5 = bitcast i8* %v4 to <16 x i32>*
  %v6 = mul nsw i32 %a1, 2
  %v7 = getelementptr inbounds i8, i8* %a0, i32 %v6
  %v8 = bitcast i8* %v7 to <16 x i32>*
  %v9 = bitcast i8* %a5 to <16 x i32>*
  %v10 = getelementptr inbounds i8, i8* %a5, i32 %a6
  %v11 = bitcast i8* %v10 to <16 x i32>*
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.vd0()
  %v13 = load <16 x i32>, <16 x i32>* %v2, align 64, !tbaa !0
  %v14 = load <16 x i32>, <16 x i32>* %v3, align 64, !tbaa !0
  %v15 = load <16 x i32>, <16 x i32>* %v5, align 64, !tbaa !0
  %v16 = load <16 x i32>, <16 x i32>* %v8, align 64, !tbaa !0
  %v17 = load i8, i8* %a3, align 1, !tbaa !0
  %v18 = getelementptr inbounds i8, i8* %a3, i32 1
  %v19 = load i8, i8* %v18, align 1, !tbaa !0
  %v20 = zext i8 %v19 to i64
  %v21 = shl nuw nsw i64 %v20, 24
  %v22 = zext i8 %v17 to i64
  %v23 = shl nuw nsw i64 %v22, 16
  %v24 = shl nuw nsw i64 %v20, 8
  %v25 = or i64 %v22, %v23
  %v26 = or i64 %v21, %v25
  %v27 = or i64 %v24, %v26
  %v28 = trunc i64 %v27 to i32
  %v29 = getelementptr inbounds i8, i8* %a3, i32 3
  %v30 = load i8, i8* %v29, align 1, !tbaa !0
  %v31 = getelementptr inbounds i8, i8* %a3, i32 4
  %v32 = load i8, i8* %v31, align 1, !tbaa !0
  %v33 = zext i8 %v32 to i64
  %v34 = shl nuw nsw i64 %v33, 24
  %v35 = zext i8 %v30 to i64
  %v36 = shl nuw nsw i64 %v35, 16
  %v37 = shl nuw nsw i64 %v33, 8
  %v38 = or i64 %v35, %v36
  %v39 = or i64 %v34, %v38
  %v40 = or i64 %v37, %v39
  %v41 = trunc i64 %v40 to i32
  %v42 = getelementptr inbounds i8, i8* %a3, i32 6
  %v43 = load i8, i8* %v42, align 1, !tbaa !0
  %v44 = getelementptr inbounds i8, i8* %a3, i32 7
  %v45 = load i8, i8* %v44, align 1, !tbaa !0
  %v46 = zext i8 %v45 to i64
  %v47 = shl nuw nsw i64 %v46, 24
  %v48 = zext i8 %v43 to i64
  %v49 = shl nuw nsw i64 %v48, 16
  %v50 = shl nuw nsw i64 %v46, 8
  %v51 = or i64 %v48, %v49
  %v52 = or i64 %v47, %v51
  %v53 = or i64 %v50, %v52
  %v54 = trunc i64 %v53 to i32
  %v55 = getelementptr inbounds i8, i8* %a3, i32 5
  %v56 = load i8, i8* %v55, align 1, !tbaa !0
  %v57 = getelementptr inbounds i8, i8* %a3, i32 2
  %v58 = load i8, i8* %v57, align 1, !tbaa !0
  %v59 = zext i8 %v58 to i64
  %v60 = shl nuw nsw i64 %v59, 24
  %v61 = zext i8 %v56 to i64
  %v62 = shl nuw nsw i64 %v61, 16
  %v63 = shl nuw nsw i64 %v59, 8
  %v64 = or i64 %v61, %v62
  %v65 = or i64 %v60, %v64
  %v66 = or i64 %v63, %v65
  %v67 = trunc i64 %v66 to i32
  %v68 = getelementptr inbounds i8, i8* %a3, i32 8
  %v69 = load i8, i8* %v68, align 1, !tbaa !0
  %v70 = zext i8 %v69 to i64
  %v71 = shl nuw nsw i64 %v70, 24
  %v72 = shl nuw nsw i64 %v70, 16
  %v73 = shl nuw nsw i64 %v70, 8
  %v74 = or i64 %v70, %v72
  %v75 = or i64 %v71, %v74
  %v76 = or i64 %v73, %v75
  %v77 = trunc i64 %v76 to i32
  %v78 = icmp sgt i32 %a2, 64
  br i1 %v78, label %b1, label %b4

b1:                                               ; preds = %b0
  %v79 = add i32 %v6, 64
  %v80 = getelementptr inbounds i8, i8* %a0, i32 %v79
  %v81 = bitcast i8* %v80 to <16 x i32>*
  %v82 = add i32 %a1, 64
  %v83 = getelementptr inbounds i8, i8* %a0, i32 %v82
  %v84 = bitcast i8* %v83 to <16 x i32>*
  %v85 = getelementptr inbounds i8, i8* %a0, i32 64
  %v86 = bitcast i8* %v85 to <16 x i32>*
  %v87 = sub i32 64, %a1
  %v88 = getelementptr inbounds i8, i8* %a0, i32 %v87
  %v89 = bitcast i8* %v88 to <16 x i32>*
  %v90 = add i32 %a2, -65
  %v91 = lshr i32 %v90, 6
  %v92 = mul i32 %v91, 64
  %v93 = add i32 %v92, %a6
  %v94 = add i32 %v93, 64
  %v95 = getelementptr i8, i8* %a5, i32 %v94
  %v96 = add i32 %v92, 64
  %v97 = getelementptr i8, i8* %a5, i32 %v96
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v98 = phi i32 [ %a2, %b1 ], [ %v153, %b2 ]
  %v99 = phi <16 x i32> [ %v12, %b1 ], [ %v100, %b2 ]
  %v100 = phi <16 x i32> [ %v13, %b1 ], [ %v118, %b2 ]
  %v101 = phi <16 x i32> [ %v12, %b1 ], [ %v102, %b2 ]
  %v102 = phi <16 x i32> [ %v14, %b1 ], [ %v120, %b2 ]
  %v103 = phi <16 x i32> [ %v12, %b1 ], [ %v104, %b2 ]
  %v104 = phi <16 x i32> [ %v15, %b1 ], [ %v122, %b2 ]
  %v105 = phi <16 x i32> [ %v12, %b1 ], [ %v106, %b2 ]
  %v106 = phi <16 x i32> [ %v16, %b1 ], [ %v124, %b2 ]
  %v107 = phi <16 x i32>* [ %v89, %b1 ], [ %v117, %b2 ]
  %v108 = phi <16 x i32>* [ %v86, %b1 ], [ %v119, %b2 ]
  %v109 = phi <16 x i32>* [ %v84, %b1 ], [ %v121, %b2 ]
  %v110 = phi <16 x i32>* [ %v81, %b1 ], [ %v123, %b2 ]
  %v111 = phi <16 x i32>* [ %v9, %b1 ], [ %v148, %b2 ]
  %v112 = phi <16 x i32>* [ %v11, %b1 ], [ %v152, %b2 ]
  %v113 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v100, <16 x i32> %v99, i32 1)
  %v114 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v102, <16 x i32> %v101, i32 1)
  %v115 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v104, <16 x i32> %v103, i32 1)
  %v116 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v106, <16 x i32> %v105, i32 1)
  %v117 = getelementptr inbounds <16 x i32>, <16 x i32>* %v107, i32 1
  %v118 = load <16 x i32>, <16 x i32>* %v107, align 64, !tbaa !0
  %v119 = getelementptr inbounds <16 x i32>, <16 x i32>* %v108, i32 1
  %v120 = load <16 x i32>, <16 x i32>* %v108, align 64, !tbaa !0
  %v121 = getelementptr inbounds <16 x i32>, <16 x i32>* %v109, i32 1
  %v122 = load <16 x i32>, <16 x i32>* %v109, align 64, !tbaa !0
  %v123 = getelementptr inbounds <16 x i32>, <16 x i32>* %v110, i32 1
  %v124 = load <16 x i32>, <16 x i32>* %v110, align 64, !tbaa !0
  %v125 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v118, <16 x i32> %v100, i32 1)
  %v126 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v120, <16 x i32> %v102, i32 1)
  %v127 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v122, <16 x i32> %v104, i32 1)
  %v128 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v124, <16 x i32> %v106, i32 1)
  %v129 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v125, <16 x i32> %v113)
  %v130 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v126, <16 x i32> %v114)
  %v131 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %v129, i32 %v28)
  %v132 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %v130, i32 %v28)
  %v133 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v127, <16 x i32> %v115)
  %v134 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v131, <32 x i32> %v130, i32 %v41)
  %v135 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v132, <32 x i32> %v133, i32 %v41)
  %v136 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v125, <16 x i32> %v126)
  %v137 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v126, <16 x i32> %v127)
  %v138 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v134, <32 x i32> %v136, i32 %v67)
  %v139 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v135, <32 x i32> %v137, i32 %v67)
  %v140 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v128, <16 x i32> %v116)
  %v141 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v138, <32 x i32> %v133, i32 %v54)
  %v142 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v139, <32 x i32> %v140, i32 %v54)
  %v143 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v141, <16 x i32> %v127, i32 %v77)
  %v144 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v142, <16 x i32> %v128, i32 %v77)
  %v145 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v143)
  %v146 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v143)
  %v147 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v145, <16 x i32> %v146, i32 %a4)
  %v148 = getelementptr inbounds <16 x i32>, <16 x i32>* %v111, i32 1
  store <16 x i32> %v147, <16 x i32>* %v111, align 64, !tbaa !0
  %v149 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v144)
  %v150 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v144)
  %v151 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v149, <16 x i32> %v150, i32 %a4)
  %v152 = getelementptr inbounds <16 x i32>, <16 x i32>* %v112, i32 1
  store <16 x i32> %v151, <16 x i32>* %v112, align 64, !tbaa !0
  %v153 = add nsw i32 %v98, -64
  %v154 = icmp sgt i32 %v153, 64
  br i1 %v154, label %b2, label %b3

b3:                                               ; preds = %b2
  %v155 = bitcast i8* %v95 to <16 x i32>*
  %v156 = bitcast i8* %v97 to <16 x i32>*
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v157 = phi <16 x i32> [ %v100, %b3 ], [ %v12, %b0 ]
  %v158 = phi <16 x i32> [ %v118, %b3 ], [ %v13, %b0 ]
  %v159 = phi <16 x i32> [ %v102, %b3 ], [ %v12, %b0 ]
  %v160 = phi <16 x i32> [ %v120, %b3 ], [ %v14, %b0 ]
  %v161 = phi <16 x i32> [ %v104, %b3 ], [ %v12, %b0 ]
  %v162 = phi <16 x i32> [ %v122, %b3 ], [ %v15, %b0 ]
  %v163 = phi <16 x i32> [ %v106, %b3 ], [ %v12, %b0 ]
  %v164 = phi <16 x i32> [ %v124, %b3 ], [ %v16, %b0 ]
  %v165 = phi <16 x i32>* [ %v156, %b3 ], [ %v9, %b0 ]
  %v166 = phi <16 x i32>* [ %v155, %b3 ], [ %v11, %b0 ]
  %v167 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v158, <16 x i32> %v157, i32 1)
  %v168 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v160, <16 x i32> %v159, i32 1)
  %v169 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v162, <16 x i32> %v161, i32 1)
  %v170 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v164, <16 x i32> %v163, i32 1)
  %v171 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v158, <16 x i32> %v158, i32 1)
  %v172 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v160, <16 x i32> %v160, i32 1)
  %v173 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v162, <16 x i32> %v162, i32 1)
  %v174 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v164, <16 x i32> %v164, i32 1)
  %v175 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v171, <16 x i32> %v167)
  %v176 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v172, <16 x i32> %v168)
  %v177 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %v175, i32 %v28)
  %v178 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %v176, i32 %v28)
  %v179 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v173, <16 x i32> %v169)
  %v180 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v177, <32 x i32> %v176, i32 %v41)
  %v181 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v178, <32 x i32> %v179, i32 %v41)
  %v182 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v171, <16 x i32> %v172)
  %v183 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v172, <16 x i32> %v173)
  %v184 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v180, <32 x i32> %v182, i32 %v67)
  %v185 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v181, <32 x i32> %v183, i32 %v67)
  %v186 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v174, <16 x i32> %v170)
  %v187 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v184, <32 x i32> %v179, i32 %v54)
  %v188 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v185, <32 x i32> %v186, i32 %v54)
  %v189 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v187, <16 x i32> %v173, i32 %v77)
  %v190 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v188, <16 x i32> %v174, i32 %v77)
  %v191 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v189)
  %v192 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v189)
  %v193 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v191, <16 x i32> %v192, i32 %a4)
  store <16 x i32> %v193, <16 x i32>* %v165, align 64, !tbaa !0
  %v194 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v190)
  %v195 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v190)
  %v196 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v194, <16 x i32> %v195, i32 %a4)
  store <16 x i32> %v196, <16 x i32>* %v166, align 64, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vd0() #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
