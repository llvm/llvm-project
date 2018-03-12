; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; REQUIRES: asserts

; Check that the code compiles successfully.
; CHECK: call f1

target triple = "hexagon-unknown--elf"

%s.0 = type { i64, i8*, [4 x i32], [4 x i32], [4 x i32], i32, i8, i8, [6 x i8] }

; Function Attrs: nounwind
declare noalias i8* @f0() local_unnamed_addr #0

; Function Attrs: nounwind
declare void @f1() local_unnamed_addr #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vlsrw.128B(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshufeh.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32>, <64 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vmpyuh.128B(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaslw.acc.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: noreturn nounwind
define void @f2(%s.0* noalias nocapture readonly %a01, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6) local_unnamed_addr #2 {
b0:
  %v0 = getelementptr inbounds %s.0, %s.0* %a01, i32 0, i32 1
  %v1 = bitcast i8** %v0 to i16**
  %v2 = load i16*, i16** %v1, align 4
  %v3 = tail call i8* @f0()
  %v4 = icmp sgt i32 %a1, 0
  %v5 = select i1 %v4, i32 0, i32 %a1
  %v6 = or i32 %v5, 1
  %v7 = icmp sgt i32 %v6, 0
  br i1 %v7, label %b1, label %b2, !prof !1

b1:                                               ; preds = %b0
  br label %b4

b2:                                               ; preds = %b0
  %v8 = ashr i32 %a6, 6
  %v9 = mul i32 %v8, 64
  %v10 = add nsw i32 %v9, 255
  %v11 = icmp sgt i32 %a6, -193
  %v12 = ashr i32 %a5, 6
  %v13 = ashr i32 %a4, 6
  %v14 = ashr i32 %a2, 6
  %v15 = icmp ult i32 %v10, 128
  %v16 = tail call i8* @f0()
  %v17 = icmp eq i8* %v16, null
  br i1 %v17, label %b6, label %b3, !prof !2

b3:                                               ; preds = %b2
  %v18 = mul nsw i32 %v13, 16
  %v19 = mul nsw i32 %v13, 19
  %v20 = mul nsw i32 %v13, 17
  %v21 = mul nsw i32 %v13, 18
  br label %b7

b4:                                               ; preds = %b4, %b1
  br label %b4

b5:                                               ; preds = %b8
  br label %b6

b6:                                               ; preds = %b5, %b2
  tail call void @f1() #3
  unreachable

b7:                                               ; preds = %b8, %b3
  %v22 = phi i8* [ %v16, %b3 ], [ %v28, %b8 ]
  %v23 = phi i32 [ 1, %b3 ], [ %v27, %b8 ]
  %v24 = sub i32 %v23, %a3
  %v25 = mul i32 %v24, %v12
  %v26 = sub i32 %v25, %v14
  br i1 %v11, label %b9, label %b8

b8:                                               ; preds = %b13, %b7
  %v27 = add nuw nsw i32 %v23, 1
  %v28 = tail call i8* @f0()
  %v29 = icmp eq i8* %v28, null
  br i1 %v29, label %b5, label %b7, !prof !2

b9:                                               ; preds = %b7
  %v30 = add i32 %v26, %v18
  %v31 = add i32 %v26, %v19
  %v32 = add i32 %v26, %v20
  %v33 = add i32 %v26, %v21
  %v34 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 undef) #3
  %v35 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 8) #3
  %v36 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v35, <32 x i32> %v35)
  %v37 = bitcast i8* %v22 to i16*
  br i1 %v15, label %b13, label %b10

b10:                                              ; preds = %b9
  %v38 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> undef) #3
  %v39 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> undef, <64 x i32> %v38) #3
  %v40 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v39, <64 x i32> %v36) #3
  %v41 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v40)
  %v42 = tail call <32 x i32> @llvm.hexagon.V6.vlsrw.128B(<32 x i32> %v41, i32 4) #3
  %v43 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> undef, <32 x i32> %v42)
  %v44 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v43) #3
  %v45 = tail call <32 x i32> @llvm.hexagon.V6.vshufeh.128B(<32 x i32> undef, <32 x i32> %v44) #3
  br label %b11

b11:                                              ; preds = %b11, %b10
  %v46 = phi <32 x i32> [ %v120, %b11 ], [ undef, %b10 ]
  %v47 = phi <32 x i32> [ %v115, %b11 ], [ undef, %b10 ]
  %v48 = phi <32 x i32> [ %v110, %b11 ], [ undef, %b10 ]
  %v49 = phi i32 [ %v124, %b11 ], [ 0, %b10 ]
  %v50 = phi i32 [ %v125, %b11 ], [ undef, %b10 ]
  %v51 = add i32 %v49, %v33
  %v52 = shl nsw i32 %v51, 6
  %v53 = getelementptr inbounds i16, i16* %v2, i32 %v52
  %v54 = bitcast i16* %v53 to <32 x i32>*
  %v55 = load <32 x i32>, <32 x i32>* %v54, align 128, !tbaa !3
  %v56 = add i32 %v49, %v32
  %v57 = shl nsw i32 %v56, 6
  %v58 = getelementptr inbounds i16, i16* %v2, i32 %v57
  %v59 = bitcast i16* %v58 to <32 x i32>*
  %v60 = load <32 x i32>, <32 x i32>* %v59, align 128, !tbaa !3
  %v61 = add i32 %v31, %v49
  %v62 = shl nsw i32 %v61, 6
  %v63 = getelementptr inbounds i16, i16* %v2, i32 %v62
  %v64 = bitcast i16* %v63 to <32 x i32>*
  %v65 = load <32 x i32>, <32 x i32>* %v64, align 128, !tbaa !3
  %v66 = add i32 %v49, %v30
  %v67 = shl nsw i32 %v66, 6
  %v68 = getelementptr inbounds i16, i16* %v2, i32 %v67
  %v69 = bitcast i16* %v68 to <32 x i32>*
  %v70 = load <32 x i32>, <32 x i32>* %v69, align 128, !tbaa !3
  %v71 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v55, <32 x i32> undef, i32 92)
  %v72 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> %v71, i32 1) #3
  %v73 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v72, <32 x i32> %v34) #3
  %v74 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.128B(<32 x i32> %v73, i32 393222) #3
  %v75 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v60, <32 x i32> %v48, i32 92)
  %v76 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> %v75, i32 1) #3
  %v77 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v76, <32 x i32> %v34) #3
  %v78 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v65, <32 x i32> undef, i32 92)
  %v79 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> %v78, i32 1) #3
  %v80 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v79, <32 x i32> %v34) #3
  %v81 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v77, <32 x i32> %v80) #3
  %v82 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %v81) #3
  %v83 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v74)
  %v84 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v82)
  %v85 = tail call <32 x i32> @llvm.hexagon.V6.vaslw.acc.128B(<32 x i32> %v83, <32 x i32> %v84, i32 2) #3
  %v86 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v85, <32 x i32> undef)
  %v87 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v70, <32 x i32> %v47, i32 92)
  %v88 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> %v87, i32 1) #3
  %v89 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v88, <32 x i32> %v34) #3
  %v90 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> undef, <32 x i32> %v46, i32 92)
  %v91 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> %v90, i32 1) #3
  %v92 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v91, <32 x i32> %v34) #3
  %v93 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v89, <32 x i32> %v92) #3
  %v94 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %v93) #3
  %v95 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v86, <64 x i32> %v94) #3
  %v96 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v95, <64 x i32> %v36) #3
  %v97 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v96)
  %v98 = tail call <32 x i32> @llvm.hexagon.V6.vlsrw.128B(<32 x i32> %v97, i32 4) #3
  %v99 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v98, <32 x i32> undef)
  %v100 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v99) #3
  %v101 = tail call <32 x i32> @llvm.hexagon.V6.vshufeh.128B(<32 x i32> undef, <32 x i32> %v100) #3
  %v102 = shl nsw i32 %v49, 6
  %v103 = getelementptr inbounds i16, i16* %v37, i32 %v102
  %v104 = bitcast i16* %v103 to <32 x i32>*
  store <32 x i32> %v101, <32 x i32>* %v104, align 128, !tbaa !6
  %v105 = or i32 %v49, 1
  %v106 = add i32 %v105, %v32
  %v107 = shl nsw i32 %v106, 6
  %v108 = getelementptr inbounds i16, i16* %v2, i32 %v107
  %v109 = bitcast i16* %v108 to <32 x i32>*
  %v110 = load <32 x i32>, <32 x i32>* %v109, align 128, !tbaa !3
  %v111 = add i32 %v105, %v30
  %v112 = shl nsw i32 %v111, 6
  %v113 = getelementptr inbounds i16, i16* %v2, i32 %v112
  %v114 = bitcast i16* %v113 to <32 x i32>*
  %v115 = load <32 x i32>, <32 x i32>* %v114, align 128, !tbaa !3
  %v116 = add i32 %v105, %v26
  %v117 = shl nsw i32 %v116, 6
  %v118 = getelementptr inbounds i16, i16* %v2, i32 %v117
  %v119 = bitcast i16* %v118 to <32 x i32>*
  %v120 = load <32 x i32>, <32 x i32>* %v119, align 128, !tbaa !3
  %v121 = shl nsw i32 %v105, 6
  %v122 = getelementptr inbounds i16, i16* %v37, i32 %v121
  %v123 = bitcast i16* %v122 to <32 x i32>*
  store <32 x i32> %v45, <32 x i32>* %v123, align 128, !tbaa !6
  %v124 = add nuw nsw i32 %v49, 2
  %v125 = add i32 %v50, -2
  %v126 = icmp eq i32 %v125, 0
  br i1 %v126, label %b12, label %b11

b12:                                              ; preds = %b11
  br label %b13

b13:                                              ; preds = %b12, %b9
  %v127 = phi i32 [ 0, %b9 ], [ %v124, %b12 ]
  %v128 = add i32 %v127, %v33
  %v129 = shl nsw i32 %v128, 6
  %v130 = getelementptr inbounds i16, i16* %v2, i32 %v129
  %v131 = bitcast i16* %v130 to <32 x i32>*
  %v132 = load <32 x i32>, <32 x i32>* %v131, align 128, !tbaa !3
  %v133 = add i32 %v127, %v30
  %v134 = shl nsw i32 %v133, 6
  %v135 = getelementptr inbounds i16, i16* %v2, i32 %v134
  %v136 = bitcast i16* %v135 to <32 x i32>*
  %v137 = load <32 x i32>, <32 x i32>* %v136, align 128, !tbaa !3
  %v138 = add i32 %v127, %v26
  %v139 = shl nsw i32 %v138, 6
  %v140 = getelementptr inbounds i16, i16* %v2, i32 %v139
  %v141 = bitcast i16* %v140 to <32 x i32>*
  %v142 = load <32 x i32>, <32 x i32>* %v141, align 128, !tbaa !3
  %v143 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v132, <32 x i32> undef, i32 92)
  %v144 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> %v143, i32 1) #3
  %v145 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v144, <32 x i32> %v34) #3
  %v146 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.128B(<32 x i32> %v145, i32 393222) #3
  %v147 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v146)
  %v148 = tail call <32 x i32> @llvm.hexagon.V6.vaslw.acc.128B(<32 x i32> %v147, <32 x i32> undef, i32 2) #3
  %v149 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v148, <32 x i32> undef)
  %v150 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v137, <32 x i32> undef, i32 92)
  %v151 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> %v150, i32 1) #3
  %v152 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v151, <32 x i32> %v34) #3
  %v153 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v142, <32 x i32> undef, i32 92)
  %v154 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> %v153, i32 1) #3
  %v155 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v154, <32 x i32> %v34) #3
  %v156 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v152, <32 x i32> %v155) #3
  %v157 = tail call <64 x i32> @llvm.hexagon.V6.vzh.128B(<32 x i32> %v156) #3
  %v158 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v149, <64 x i32> %v157) #3
  %v159 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v158, <64 x i32> %v36) #3
  %v160 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v159)
  %v161 = tail call <32 x i32> @llvm.hexagon.V6.vlsrw.128B(<32 x i32> %v160, i32 4) #3
  %v162 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v161, <32 x i32> undef)
  %v163 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v162) #3
  %v164 = tail call <32 x i32> @llvm.hexagon.V6.vshufeh.128B(<32 x i32> %v163, <32 x i32> undef) #3
  %v165 = getelementptr inbounds i16, i16* %v37, i32 undef
  %v166 = bitcast i16* %v165 to <32 x i32>*
  store <32 x i32> %v164, <32 x i32>* %v166, align 128, !tbaa !6
  br label %b8
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
attributes #2 = { noreturn nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"halide_mattrs", !"+hvxv60,+hvx-length128b"}
!1 = !{!"branch_weights", i32 1073741824, i32 0}
!2 = !{!"branch_weights", i32 0, i32 1073741824}
!3 = !{!4, !4, i64 0}
!4 = !{!"input_yuv", !5}
!5 = !{!"Halide buffer"}
!6 = !{!7, !7, i64 0}
!7 = !{!"blurred_ds_y", !5}
