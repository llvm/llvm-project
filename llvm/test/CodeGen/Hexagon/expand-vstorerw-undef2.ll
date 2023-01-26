; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Dead defs may still appear live in LivePhysRegs, leading to an expansion
; of a double-vector store that uses an undefined source register.

target triple = "hexagon-unknown--elf"

declare noalias ptr @halide_malloc() local_unnamed_addr #0
declare void @halide_free() local_unnamed_addr #0

declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32>, <32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32>, <32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vavghrnd.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vlsrw.128B(<32 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vpackeh.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vshufoh.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vsubhsat.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vaddhw.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32>, <64 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vmpyuh.128B(<32 x i32>, i32) #1
declare <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32>, <32 x i32>, i32) #1
declare <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32>, <32 x i32>, i32) #1

define hidden void @fred() #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  ret void

b2:                                               ; preds = %b0
  %v3 = tail call ptr @halide_malloc()
  %v5 = tail call ptr @halide_malloc()
  %v7 = tail call ptr @halide_malloc()
  %v9 = tail call ptr @halide_malloc()
  br label %b11

b11:                                              ; preds = %b11, %b2
  br i1 undef, label %b12, label %b11

b12:                                              ; preds = %b11
  br i1 undef, label %b16, label %b13

b13:                                              ; preds = %b12
  %v14 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> zeroinitializer) #2
  %v15 = tail call <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32> undef, <32 x i32> %v14, i32 1) #2
  br i1 undef, label %b19, label %b17

b16:                                              ; preds = %b12
  unreachable

b17:                                              ; preds = %b13
  %v18 = tail call <32 x i32> @llvm.hexagon.V6.vavghrnd.128B(<32 x i32> %v15, <32 x i32> undef) #2
  br label %b19

b19:                                              ; preds = %b17, %b13
  %v20 = phi <32 x i32> [ %v18, %b17 ], [ %v15, %b13 ]
  %v21 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> zeroinitializer, <32 x i32> %v20) #2
  %v22 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %v21, <32 x i32> undef, i32 -2)
  %v23 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v22)
  store <32 x i32> %v23, ptr undef, align 128
  tail call void @halide_free() #3
  br label %b24

b24:                                              ; preds = %b33, %b19
  %v25 = load <32 x i32>, ptr undef, align 128
  %v26 = fptoui float undef to i16
  %v27 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 -2147450880) #2
  %v28 = xor i16 %v26, -1
  %v29 = zext i16 %v28 to i32
  %v30 = or i32 0, %v29
  %v31 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 1) #2
  %v32 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v31, <32 x i32> %v31)
  br label %b34

b33:                                              ; preds = %b34
  br label %b24

b34:                                              ; preds = %b34, %b24
  %v35 = phi <32 x i32> [ %v45, %b34 ], [ undef, %b24 ]
  %v36 = phi <32 x i32> [ undef, %b34 ], [ %v25, %b24 ]
  %v37 = phi <32 x i32> [ %v46, %b34 ], [ undef, %b24 ]
  %v38 = phi i32 [ %v145, %b34 ], [ 0, %b24 ]
  %v39 = load <32 x i32>, ptr undef, align 128
  %v40 = add nsw i32 %v38, undef
  %v41 = shl nsw i32 %v40, 6
  %v42 = add nsw i32 %v41, 64
  %v43 = getelementptr inbounds i16, ptr %v5, i32 %v42
  %v45 = load <32 x i32>, ptr %v43, align 128
  %v46 = load <32 x i32>, ptr undef, align 128
  %v47 = load <32 x i32>, ptr null, align 128
  %v48 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> undef, <32 x i32> undef, i32 2)
  %v49 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v45, <32 x i32> %v35, i32 24)
  %v50 = tail call <32 x i32> @llvm.hexagon.V6.vsubhsat.128B(<32 x i32> %v48, <32 x i32> %v49) #2
  %v51 = tail call <64 x i32> @llvm.hexagon.V6.vaddhw.128B(<32 x i32> undef, <32 x i32> %v50) #2
  %v52 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v39, <32 x i32> %v47, i32 50)
  %v53 = tail call <32 x i32> @llvm.hexagon.V6.vpackeh.128B(<32 x i32> %v52, <32 x i32> undef)
  %v54 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v53, <32 x i32> %v27) #2
  %v55 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> undef, <32 x i32> %v54, i32 undef) #2
  %v56 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v55, <64 x i32> zeroinitializer) #2
  %v57 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v56)
  %v58 = tail call <32 x i32> @llvm.hexagon.V6.vlsrw.128B(<32 x i32> %v57, i32 16) #2
  %v59 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v56)
  %v60 = tail call <32 x i32> @llvm.hexagon.V6.vlsrw.128B(<32 x i32> %v59, i32 16) #2
  %v61 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v60, <32 x i32> %v58)
  %v62 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v61, <64 x i32> %v55) #2
  %v63 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v62, <64 x i32> zeroinitializer) #2
  %v64 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v63) #2
  %v65 = tail call <32 x i32> @llvm.hexagon.V6.vshufoh.128B(<32 x i32> %v64, <32 x i32> undef) #2
  %v66 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v65, <32 x i32> %v27) #2
  %v67 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v66, <32 x i32> undef) #2
  %v68 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> zeroinitializer, <32 x i32> %v27) #2
  %v69 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.128B(<32 x i32> %v68, i32 %v30) #2
  %v70 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v47, <32 x i32> undef, i32 52)
  %v71 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v39, <32 x i32> %v47, i32 52)
  %v72 = tail call <32 x i32> @llvm.hexagon.V6.vpackeh.128B(<32 x i32> %v71, <32 x i32> %v70)
  %v73 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v72, <32 x i32> %v27) #2
  %v74 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuh.acc.128B(<64 x i32> %v69, <32 x i32> %v73, i32 undef) #2
  %v75 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v74, <64 x i32> zeroinitializer) #2
  %v76 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v75)
  %v77 = tail call <32 x i32> @llvm.hexagon.V6.vlsrw.128B(<32 x i32> %v76, i32 16) #2
  %v78 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> undef, <32 x i32> %v77)
  %v79 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v78, <64 x i32> %v74) #2
  %v80 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v79, <64 x i32> zeroinitializer) #2
  %v81 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v80) #2
  %v82 = tail call <32 x i32> @llvm.hexagon.V6.vshufoh.128B(<32 x i32> %v81, <32 x i32> undef) #2
  %v83 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v82, <32 x i32> %v27) #2
  %v84 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v51, <64 x i32> %v32) #2
  %v85 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v84) #2
  %v86 = tail call <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32> undef, <32 x i32> %v85, i32 1) #2
  %v87 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v83, <32 x i32> %v86) #2
  %v88 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %v87, <32 x i32> %v67, i32 -2)
  %v89 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v88)
  %v90 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v88)
  %v91 = getelementptr inbounds i16, ptr %v9, i32 undef
  store <32 x i32> %v90, ptr %v91, align 128
  %v93 = getelementptr inbounds i16, ptr %v9, i32 undef
  store <32 x i32> %v89, ptr %v93, align 128
  %v95 = getelementptr inbounds i16, ptr %v3, i32 undef
  %v97 = load <32 x i32>, ptr %v95, align 128
  %v98 = getelementptr inbounds i16, ptr %v7, i32 undef
  %v100 = load <32 x i32>, ptr %v98, align 128
  %v101 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> undef, <32 x i32> %v36, i32 22)
  %v102 = tail call <32 x i32> @llvm.hexagon.V6.vsubhsat.128B(<32 x i32> %v100, <32 x i32> %v101) #2
  %v103 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> undef, <32 x i32> %v102) #2
  %v104 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v97, <32 x i32> %v37, i32 48)
  %v105 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v46, <32 x i32> %v97, i32 48)
  %v106 = tail call <32 x i32> @llvm.hexagon.V6.vpackeh.128B(<32 x i32> %v105, <32 x i32> %v104)
  %v107 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> undef, <64 x i32> %v32) #2
  %v108 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v107) #2
  %v109 = tail call <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32> undef, <32 x i32> %v108, i32 1) #2
  %v110 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v106, <32 x i32> %v109) #2
  %v111 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %v110, <32 x i32> %v103, i32 -2)
  %v112 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v111)
  %v113 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v111)
  %v114 = getelementptr inbounds i16, ptr %v9, i32 undef
  store <32 x i32> %v113, ptr %v114, align 128
  %v116 = getelementptr inbounds i16, ptr %v9, i32 undef
  store <32 x i32> %v112, ptr %v116, align 128
  %v118 = getelementptr inbounds i16, ptr %v3, i32 undef
  %v120 = load <32 x i32>, ptr %v118, align 128
  %v121 = getelementptr inbounds i16, ptr %v5, i32 undef
  %v123 = load <32 x i32>, ptr %v121, align 128
  %v126 = load <32 x i32>, ptr %v5, align 128
  %v127 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v126, <32 x i32> %v123, i32 22)
  %v128 = tail call <32 x i32> @llvm.hexagon.V6.vsubhsat.128B(<32 x i32> undef, <32 x i32> %v127) #2
  %v129 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v126, <32 x i32> %v123, i32 24)
  %v130 = tail call <32 x i32> @llvm.hexagon.V6.vsubhsat.128B(<32 x i32> undef, <32 x i32> %v129) #2
  %v131 = tail call <64 x i32> @llvm.hexagon.V6.vaddhw.128B(<32 x i32> %v128, <32 x i32> %v130) #2
  %v132 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v120, <32 x i32> undef, i32 46)
  %v133 = tail call <32 x i32> @llvm.hexagon.V6.vpackeh.128B(<32 x i32> undef, <32 x i32> %v132)
  %v134 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v133, <32 x i32> %v128) #2
  %v135 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> %v120, <32 x i32> undef, i32 48)
  %v136 = tail call <32 x i32> @llvm.hexagon.V6.valignb.128B(<32 x i32> undef, <32 x i32> %v120, i32 48)
  %v137 = tail call <32 x i32> @llvm.hexagon.V6.vpackeh.128B(<32 x i32> %v136, <32 x i32> %v135)
  %v138 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v131, <64 x i32> %v32) #2
  %v139 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v138) #2
  %v140 = tail call <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32> %v139, <32 x i32> undef, i32 1) #2
  %v141 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v137, <32 x i32> %v140) #2
  %v142 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %v141, <32 x i32> %v134, i32 -2)
  %v143 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v142)
  %v144 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v142)
  store <32 x i32> %v144, ptr undef, align 128
  store <32 x i32> %v143, ptr undef, align 128
  %v145 = add nuw nsw i32 %v38, 1
  %v146 = icmp eq i32 %v38, undef
  br i1 %v146, label %b33, label %b34
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { nobuiltin nounwind }
