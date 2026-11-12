; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; Test that the register scavenger does not assert because a spill slot
; was not found. The bug is that the Hexagon spill code was not allocating
; the spill slot because the function that returns true, which indicates
; the code changed when a spill is inserted, was not always returning true.

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, i32 %a1, i32 %a2, i32 %a3, ptr noalias nocapture %a4, i32 %a5) #0 {
b0:
  %v0 = sub i32 0, %a1
  %v1 = getelementptr inbounds i8, ptr %a0, i32 %v0
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %a1
  %v3 = mul nsw i32 %a1, 2
  %v4 = getelementptr inbounds i8, ptr %a0, i32 %v3
  %v6 = getelementptr inbounds i8, ptr %a4, i32 %a5
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.vd0()
  %v9 = load <16 x i32>, ptr undef, align 64
  %v10 = or i64 undef, 0
  %v11 = trunc i64 %v10 to i32
  %v12 = load i8, ptr undef, align 1
  %v13 = zext i8 %v12 to i64
  %v14 = shl nuw nsw i64 %v13, 8
  %v15 = or i64 0, %v14
  %v16 = trunc i64 %v15 to i32
  %v17 = load i8, ptr undef, align 1
  %v18 = zext i8 %v17 to i64
  %v19 = or i64 0, %v18
  %v20 = or i64 %v19, 0
  %v21 = or i64 %v20, 0
  %v22 = trunc i64 %v21 to i32
  %v23 = load i8, ptr undef, align 1
  %v24 = zext i8 %v23 to i64
  %v25 = shl nuw nsw i64 %v24, 8
  %v26 = or i64 undef, %v25
  %v27 = trunc i64 %v26 to i32
  %v28 = icmp sgt i32 %a2, 64
  br i1 %v28, label %b1, label %b6

b1:                                               ; preds = %b0
  %v29 = getelementptr inbounds i8, ptr %v4, i32 64
  %v31 = getelementptr inbounds i8, ptr %v2, i32 64
  %v33 = getelementptr inbounds i8, ptr %a0, i32 64
  %v35 = getelementptr inbounds i8, ptr %v1, i32 64
  %v37 = add i32 0, 64
  %v38 = getelementptr i8, ptr %a4, i32 %v37
  %v39 = add i32 %a2, -65
  %v40 = lshr i32 %v39, 6
  %v41 = add nuw nsw i32 %v40, 1
  %v42 = and i32 %v41, 3
  %v43 = icmp eq i32 %v42, 0
  br i1 undef, label %b2, label %b4

b2:                                               ; preds = %b2, %b1
  %v44 = phi i32 [ %v144, %b2 ], [ %a2, %b1 ]
  %v45 = phi <16 x i32> [ %v101, %b2 ], [ %v8, %b1 ]
  %v46 = phi <16 x i32> [ %v113, %b2 ], [ undef, %b1 ]
  %v47 = phi <16 x i32> [ %v102, %b2 ], [ %v8, %b1 ]
  %v48 = phi <16 x i32> [ %v118, %b2 ], [ undef, %b1 ]
  %v49 = phi ptr [ %v112, %b2 ], [ %v35, %b1 ]
  %v50 = phi ptr [ %v114, %b2 ], [ %v33, %b1 ]
  %v51 = phi ptr [ %v116, %b2 ], [ %v31, %b1 ]
  %v52 = phi ptr [ undef, %b2 ], [ %v29, %b1 ]
  %v53 = phi ptr [ %v139, %b2 ], [ %a4, %b1 ]
  %v54 = phi ptr [ %v143, %b2 ], [ %v6, %b1 ]
  %v55 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v46, <16 x i32> %v45, i32 1)
  %v56 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> undef, <16 x i32> %v47, i32 1)
  %v57 = getelementptr inbounds <16 x i32>, ptr %v49, i32 1
  %v58 = load <16 x i32>, ptr %v49, align 64
  %v59 = getelementptr inbounds <16 x i32>, ptr %v50, i32 1
  %v60 = load <16 x i32>, ptr %v50, align 64
  %v61 = load <16 x i32>, ptr %v51, align 64
  %v62 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v58, <16 x i32> %v46, i32 1)
  %v63 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v60, <16 x i32> undef, i32 1)
  %v64 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v61, <16 x i32> undef, i32 1)
  %v65 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> undef, <16 x i32> %v48, i32 1)
  %v66 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v62, <16 x i32> %v55)
  %v67 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v63, <16 x i32> %v56)
  %v68 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %v66, i32 %v11)
  %v69 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v64, <16 x i32> undef)
  %v70 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v68, <32 x i32> %v67, i32 %v16)
  %v71 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v62, <16 x i32> %v63)
  %v72 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v70, <32 x i32> %v71, i32 %v22)
  %v73 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v72, <32 x i32> %v69, i32 0)
  %v74 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v73, <16 x i32> %v64, i32 %v27)
  %v75 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> zeroinitializer, <16 x i32> %v65, i32 %v27)
  %v76 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v74)
  %v77 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v76, <16 x i32> undef, i32 %a3)
  %v78 = getelementptr inbounds <16 x i32>, ptr %v53, i32 1
  store <16 x i32> %v77, ptr %v53, align 64
  %v79 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v75)
  %v80 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v79, <16 x i32> undef, i32 %a3)
  %v81 = getelementptr inbounds <16 x i32>, ptr %v54, i32 1
  store <16 x i32> %v80, ptr %v54, align 64
  %v82 = getelementptr inbounds <16 x i32>, ptr %v49, i32 2
  %v83 = load <16 x i32>, ptr %v57, align 64
  %v84 = getelementptr inbounds <16 x i32>, ptr %v50, i32 2
  %v85 = load <16 x i32>, ptr %v59, align 64
  %v86 = load <16 x i32>, ptr undef, align 64
  %v87 = load <16 x i32>, ptr null, align 64
  %v88 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v83, <16 x i32> %v58, i32 1)
  %v89 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v85, <16 x i32> %v60, i32 1)
  %v90 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v86, <16 x i32> %v61, i32 1)
  %v91 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v90, <16 x i32> undef)
  %v92 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> undef, <32 x i32> undef, i32 %v16)
  %v93 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v88, <16 x i32> %v89)
  %v94 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v92, <32 x i32> %v93, i32 %v22)
  %v95 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v94, <32 x i32> %v91, i32 0)
  %v96 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v95, <16 x i32> %v90, i32 %v27)
  %v97 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v96)
  %v98 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v97, <16 x i32> undef, i32 %a3)
  store <16 x i32> %v98, ptr %v78, align 64
  %v99 = getelementptr inbounds <16 x i32>, ptr %v54, i32 2
  store <16 x i32> undef, ptr %v81, align 64
  %v100 = getelementptr inbounds <16 x i32>, ptr %v49, i32 3
  %v101 = load <16 x i32>, ptr %v82, align 64
  %v102 = load <16 x i32>, ptr %v84, align 64
  %v103 = getelementptr inbounds <16 x i32>, ptr %v51, i32 3
  %v104 = load <16 x i32>, ptr null, align 64
  %v105 = getelementptr inbounds <16 x i32>, ptr %v52, i32 3
  %v106 = load <16 x i32>, ptr undef, align 64
  %v107 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> undef, <16 x i32> undef, i32 %a3)
  store <16 x i32> %v107, ptr undef, align 64
  %v108 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> undef, <16 x i32> undef, i32 %a3)
  %v109 = getelementptr inbounds <16 x i32>, ptr %v54, i32 3
  store <16 x i32> %v108, ptr %v99, align 64
  %v110 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v104, <16 x i32> %v86, i32 1)
  %v111 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v106, <16 x i32> %v87, i32 1)
  %v112 = getelementptr inbounds <16 x i32>, ptr %v49, i32 4
  %v113 = load <16 x i32>, ptr %v100, align 64
  %v114 = getelementptr inbounds <16 x i32>, ptr %v50, i32 4
  %v115 = load <16 x i32>, ptr undef, align 64
  %v116 = getelementptr inbounds <16 x i32>, ptr %v51, i32 4
  %v117 = load <16 x i32>, ptr %v103, align 64
  %v118 = load <16 x i32>, ptr %v105, align 64
  %v119 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v113, <16 x i32> %v101, i32 1)
  %v120 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v115, <16 x i32> %v102, i32 1)
  %v121 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v117, <16 x i32> %v104, i32 1)
  %v122 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v118, <16 x i32> %v106, i32 1)
  %v123 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v119, <16 x i32> undef)
  %v124 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v120, <16 x i32> undef)
  %v125 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %v123, i32 %v11)
  %v126 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %v124, i32 %v11)
  %v127 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v121, <16 x i32> %v110)
  %v128 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v125, <32 x i32> %v124, i32 %v16)
  %v129 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v126, <32 x i32> %v127, i32 %v16)
  %v130 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v128, <32 x i32> undef, i32 %v22)
  %v131 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v129, <32 x i32> undef, i32 %v22)
  %v132 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v122, <16 x i32> %v111)
  %v133 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v130, <32 x i32> %v127, i32 0)
  %v134 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v131, <32 x i32> %v132, i32 0)
  %v135 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v133, <16 x i32> %v121, i32 %v27)
  %v136 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v134, <16 x i32> %v122, i32 %v27)
  %v137 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v135)
  %v138 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v137, <16 x i32> undef, i32 %a3)
  %v139 = getelementptr inbounds <16 x i32>, ptr %v53, i32 4
  store <16 x i32> %v138, ptr undef, align 64
  %v140 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v136)
  %v141 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v136)
  %v142 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v140, <16 x i32> %v141, i32 %a3)
  %v143 = getelementptr inbounds <16 x i32>, ptr %v54, i32 4
  store <16 x i32> %v142, ptr %v109, align 64
  %v144 = add nsw i32 %v44, -256
  %v145 = icmp sgt i32 %v144, 256
  br i1 %v145, label %b2, label %b3

b3:                                               ; preds = %b2
  %v146 = phi ptr [ %v116, %b2 ]
  %v147 = phi ptr [ %v114, %b2 ]
  %v148 = phi ptr [ %v112, %b2 ]
  br i1 %v43, label %b5, label %b4

b4:                                               ; preds = %b3, %b1
  %v149 = phi <16 x i32> [ %v9, %b1 ], [ undef, %b3 ]
  %v150 = phi ptr [ %v35, %b1 ], [ %v148, %b3 ]
  %v151 = phi ptr [ %v33, %b1 ], [ %v147, %b3 ]
  %v152 = phi ptr [ %v31, %b1 ], [ %v146, %b3 ]
  %v153 = phi ptr [ %a4, %b1 ], [ undef, %b3 ]
  %v154 = load <16 x i32>, ptr %v150, align 64
  %v155 = load <16 x i32>, ptr %v151, align 64
  %v156 = load <16 x i32>, ptr %v152, align 64
  %v157 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v154, <16 x i32> undef, i32 1)
  %v158 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v155, <16 x i32> undef, i32 1)
  %v159 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v156, <16 x i32> %v149, i32 1)
  %v160 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v157, <16 x i32> %v158)
  %v161 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> undef, <32 x i32> %v160, i32 %v22)
  %v162 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v161, <32 x i32> undef, i32 0)
  %v163 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v162, <16 x i32> %v159, i32 %v27)
  %v164 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v163)
  %v165 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %v164, <16 x i32> undef, i32 %a3)
  store <16 x i32> %v165, ptr %v153, align 64
  unreachable

b5:                                               ; preds = %b3
  br label %b6

b6:                                               ; preds = %b5, %b0
  %v167 = phi <16 x i32> [ %v8, %b0 ], [ undef, %b5 ]
  %v168 = phi ptr [ %a4, %b0 ], [ %v38, %b5 ]
  %v169 = phi ptr [ %v6, %b0 ], [ undef, %b5 ]
  %v170 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> undef, <16 x i32> %v167, i32 1)
  %v171 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> undef, <16 x i32> undef, i32 1)
  %v172 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> undef, <16 x i32> %v170)
  %v173 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> undef, <16 x i32> %v171)
  %v174 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v171, <16 x i32> undef)
  %v175 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> undef, <32 x i32> %v173, i32 %v22)
  %v176 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> undef, <32 x i32> %v174, i32 %v22)
  %v177 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v175, <32 x i32> %v172, i32 0)
  %v178 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %v176, <32 x i32> undef, i32 0)
  %v179 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v177, <16 x i32> undef, i32 %v27)
  %v180 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v178, <16 x i32> undef, i32 %v27)
  %v181 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v179)
  %v182 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> undef, <16 x i32> %v181, i32 %a3)
  store <16 x i32> %v182, ptr %v168, align 64
  %v183 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v180)
  %v184 = tail call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> undef, <16 x i32> %v183, i32 %a3)
  store <16 x i32> %v184, ptr %v169, align 64
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

attributes #0 = { nounwind "target-cpu"="hexagonv62" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { nounwind readnone }
