; RUN: llc -O3 -march=hexagon < %s | FileCheck %s
; CHECK: v{{[0-9]+}}.w = vadd

target triple = "hexagon"

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind
define void @f0(i16* noalias nocapture %a0, i32* noalias nocapture readonly %a1, i32 %a2, i8* noalias nocapture readonly %a3, i1 %cond) #1 {
b0:
  %v0 = add nsw i32 %a2, 63
  %v1 = ashr i32 %v0, 6
  %v2 = bitcast i16* %a0 to <16 x i32>*
  %v3 = bitcast i8* %a3 to <16 x i32>*
  %v4 = getelementptr inbounds i32, i32* %a1, i32 32
  %v5 = bitcast i32* %v4 to <16 x i32>*
  %v6 = load <16 x i32>, <16 x i32>* %v5, align 64, !tbaa !0
  %v7 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 32768)
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 2147450879)
  %v9 = icmp sgt i32 %v1, 0
  br i1 %v9, label %b1, label %b4

b1:                                               ; preds = %b0
  %v10 = bitcast i32* %a1 to <16 x i32>*
  %v11 = load <16 x i32>, <16 x i32>* %v10, align 64, !tbaa !0
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v6, <16 x i32> %v11, i32 2)
  %v13 = getelementptr inbounds i32, i32* %a1, i32 48
  %v14 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v12, <16 x i32> undef)
  %v15 = bitcast i32* %v13 to <16 x i32>*
  br i1 %cond, label %b2, label %b3

b2:                                               ; preds = %b1
  %v16 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 1
  %v17 = load <16 x i32>, <16 x i32>* %v16, align 64, !tbaa !0
  %v18 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v17, <16 x i32> %v6, i32 4)
  %v19 = load <16 x i32>, <16 x i32>* %v15, align 64, !tbaa !0
  %v20 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 2
  %v21 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v18, <16 x i32> %v19)
  %v22 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v21, <16 x i32> %v14, i32 4)
  %v23 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v21, <16 x i32> %v14, i32 8)
  %v24 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v21, <16 x i32> %v14, i32 12)
  %v25 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v14, <16 x i32> %v22)
  %v26 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v25, <16 x i32> %v23)
  %v27 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v26, <16 x i32> %v24)
  %v28 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v19, <16 x i32> undef, i32 16)
  %v29 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v27, <16 x i32> %v11)
  %v30 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v27, <16 x i32> %v28)
  %v31 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v29, i32 53019433)
  %v32 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v30, i32 53019433)
  %v33 = load <16 x i32>, <16 x i32>* %v3, align 64, !tbaa !0
  %v34 = tail call <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32> %v33)
  %v35 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> %v34, <16 x i32> %v34)
  %v36 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %v32, <16 x i32> %v31)
  store <16 x i32> %v36, <16 x i32>* %v2, align 64, !tbaa !0
  %v37 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 3
  %v38 = load <16 x i32>, <16 x i32>* %v37, align 64, !tbaa !0
  %v39 = load <16 x i32>, <16 x i32>* %v20, align 64, !tbaa !0
  %v40 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 4
  %v41 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> undef, <16 x i32> %v39)
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v41, <16 x i32> %v21, i32 4)
  %v43 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v41, <16 x i32> %v21, i32 8)
  %v44 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v41, <16 x i32> %v21, i32 12)
  %v45 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v21, <16 x i32> %v42)
  %v46 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v45, <16 x i32> %v43)
  %v47 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v46, <16 x i32> %v44)
  %v48 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v47, <16 x i32> %v6)
  %v49 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v47, <16 x i32> undef)
  %v50 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v48, i32 53019433)
  %v51 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v49, i32 53019433)
  %v52 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %v51, <16 x i32> %v50)
  %v53 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v52, <16 x i32> undef, i32 56)
  %v54 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v35)
  %v55 = tail call <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32> %v53, <16 x i32> %v54)
  %v56 = tail call <16 x i32> @llvm.hexagon.V6.vminuh(<16 x i32> %v55, <16 x i32> %v8)
  %v57 = getelementptr inbounds <16 x i32>, <16 x i32>* %v2, i32 undef
  store <16 x i32> %v56, <16 x i32>* %v57, align 64, !tbaa !0
  %v58 = getelementptr <16 x i32>, <16 x i32>* %v2, i32 2
  %v59 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 5
  %v60 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> zeroinitializer, <16 x i32> %v38, i32 4)
  %v61 = load <16 x i32>, <16 x i32>* %v40, align 64, !tbaa !0
  %v62 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v60, <16 x i32> %v61)
  %v63 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v62, <16 x i32> %v41, i32 4)
  %v64 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v62, <16 x i32> %v41, i32 8)
  %v65 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v62, <16 x i32> %v41, i32 12)
  %v66 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v41, <16 x i32> %v63)
  %v67 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v66, <16 x i32> %v64)
  %v68 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v67, <16 x i32> %v65)
  %v69 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v61, <16 x i32> %v39, i32 16)
  %v70 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 1
  %v71 = load <16 x i32>, <16 x i32>* %v70, align 64, !tbaa !0
  %v72 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v68, <16 x i32> %v71)
  %v73 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v68, <16 x i32> %v69)
  %v74 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v72, i32 53019433)
  %v75 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v73, i32 53019433)
  %v76 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %v75, <16 x i32> %v74)
  store <16 x i32> %v76, <16 x i32>* %v58, align 64, !tbaa !0
  %v77 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 7
  %v78 = load <16 x i32>, <16 x i32>* %v77, align 64, !tbaa !0
  %v79 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> undef, <16 x i32> undef)
  %v80 = getelementptr <16 x i32>, <16 x i32>* %v2, i32 4
  %v81 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 9
  %v82 = load <16 x i32>, <16 x i32>* %v81, align 64, !tbaa !0
  %v83 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v82, <16 x i32> %v78, i32 4)
  %v84 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 10
  %v85 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v83, <16 x i32> undef)
  %v86 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v85, <16 x i32> %v79, i32 4)
  %v87 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v85, <16 x i32> %v79, i32 8)
  %v88 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v85, <16 x i32> %v79, i32 12)
  %v89 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v79, <16 x i32> %v86)
  %v90 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v89, <16 x i32> %v87)
  %v91 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v90, <16 x i32> %v88)
  %v92 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> undef, <16 x i32> undef, i32 16)
  %v93 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v91, <16 x i32> zeroinitializer)
  %v94 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v91, <16 x i32> %v92)
  %v95 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v93, i32 53019433)
  %v96 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v94, i32 53019433)
  %v97 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> undef, <16 x i32> undef)
  %v98 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %v96, <16 x i32> %v95)
  store <16 x i32> %v98, <16 x i32>* %v80, align 64, !tbaa !0
  %v99 = getelementptr inbounds <16 x i32>, <16 x i32>* %v15, i32 11
  %v100 = load <16 x i32>, <16 x i32>* %v99, align 64, !tbaa !0
  %v101 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v100, <16 x i32> %v82, i32 4)
  %v102 = load <16 x i32>, <16 x i32>* %v84, align 64, !tbaa !0
  %v103 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v101, <16 x i32> %v102)
  %v104 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v103, <16 x i32> %v85, i32 4)
  %v105 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v103, <16 x i32> %v85, i32 8)
  %v106 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v85, <16 x i32> %v104)
  %v107 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v106, <16 x i32> %v105)
  %v108 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v107, <16 x i32> undef)
  %v109 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v102, <16 x i32> undef, i32 16)
  %v110 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v108, <16 x i32> %v78)
  %v111 = tail call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %v108, <16 x i32> %v109)
  %v112 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v110, i32 53019433)
  %v113 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %v7, <16 x i32> %v111, i32 53019433)
  %v114 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %v113, <16 x i32> %v112)
  %v115 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v114, <16 x i32> undef, i32 56)
  %v116 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v97)
  %v117 = tail call <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32> %v115, <16 x i32> %v116)
  %v118 = tail call <16 x i32> @llvm.hexagon.V6.vminuh(<16 x i32> %v117, <16 x i32> %v8)
  %v119 = getelementptr inbounds <16 x i32>, <16 x i32>* %v2, i32 undef
  store <16 x i32> %v118, <16 x i32>* %v119, align 64, !tbaa !0
  %v120 = getelementptr <16 x i32>, <16 x i32>* %v2, i32 6
  %v121 = tail call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> undef, <16 x i32> undef)
  store <16 x i32> %v121, <16 x i32>* %v120, align 64, !tbaa !0
  unreachable

b3:                                               ; preds = %b1
  unreachable

b4:                                               ; preds = %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vminuh(<16 x i32>, <16 x i32>) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
