; RUN: llc -mtriple=hexagon -disable-hsdr < %s | FileCheck %s

; Check if instruction vandqrt.acc and its predecessor are scheduled in consecutive packets.
; CHECK: or(q{{[0-3]+}},q{{[0-3]+}})
; CHECK: }
; CHECK-NOT: }
; CHECK: |= vand(q{{[0-3]+}},r{{[0-9]+}})
; CHECK: endloop0

target triple = "hexagon-unknown-linux-gnu"

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, i32 %a1, i32 %a2, i32 %a3, ptr noalias nocapture %a4, i32 %a5) #0 {
b0:
  %v0 = mul i32 %a2, 3
  %v2 = mul i32 %a5, -2
  %v3 = add i32 %v2, %a1
  %v4 = and i32 %a5, 63
  %v5 = add i32 %v3, %v4
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 -1)
  %v7 = lshr i32 %v5, 6
  %v8 = and i32 %v7, 7
  %v9 = and i32 %v5, 511
  %v10 = icmp eq i32 %v9, 0
  %v11 = shl i32 -1, %v8
  %v12 = select i1 %v10, i32 0, i32 %v11
  %v13 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %v12)
  %v14 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v13)
  %v15 = tail call <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32> %v14)
  %v16 = tail call <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %v5)
  %v17 = shl i32 1, %v8
  %v18 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %v17)
  %v19 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32> %v15, <64 x i1> %v16, i32 %v18)
  %v20 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %a3)
  %v21 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v20)
  %v22 = icmp sgt i32 %v5, 0
  br i1 %v22, label %b1, label %b8

b1:                                               ; preds = %b0
  %v23 = getelementptr inbounds i8, ptr %a0, i32 %a5
  %v25 = load <16 x i32>, ptr %v23, align 64, !tbaa !0
  %v26 = add i32 %a5, 64
  %v27 = getelementptr inbounds i8, ptr %a0, i32 %v26
  %v29 = add i32 %a5, -64
  %v30 = getelementptr inbounds i8, ptr %a0, i32 %v29
  %v32 = load <16 x i32>, ptr %v30, align 64, !tbaa !0
  %v33 = tail call <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32 %a5)
  %v34 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1> %v33, i32 16843009)
  %v35 = tail call <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32> %v34)
  %v36 = add i32 %v0, %a5
  %v37 = getelementptr inbounds i8, ptr %a0, i32 %v36
  %v39 = sub i32 %a5, %v0
  %v40 = getelementptr inbounds i8, ptr %a0, i32 %v39
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.vd0()
  %v43 = add i32 %v4, %a1
  %v44 = mul i32 %a5, 2
  %v45 = sub i32 %v43, %v44
  %v46 = xor i32 %v45, -1
  %v47 = icmp sgt i32 %v46, -513
  %v48 = select i1 %v47, i32 %v46, i32 -513
  %v49 = add i32 %v48, %a1
  %v50 = add i32 %v49, %v4
  %v51 = add i32 %v50, 512
  %v52 = sub i32 %v51, %v44
  %v53 = lshr i32 %v52, 9
  %v54 = mul nuw nsw i32 %v53, 16
  %v55 = add nuw nsw i32 %v54, 16
  %v56 = getelementptr i32, ptr %a4, i32 %v55
  br label %b2

b2:                                               ; preds = %b6, %b1
  %v57 = phi i32 [ %v46, %b1 ], [ %v125, %b6 ]
  %v58 = phi i32 [ %v5, %b1 ], [ %v123, %b6 ]
  %v59 = phi ptr [ %a4, %b1 ], [ %v122, %b6 ]
  %v60 = phi ptr [ %v37, %b1 ], [ %v114, %b6 ]
  %v61 = phi ptr [ %v40, %b1 ], [ %v115, %b6 ]
  %v62 = phi ptr [ %v27, %b1 ], [ %v116, %b6 ]
  %v63 = phi i32 [ 512, %b1 ], [ %v69, %b6 ]
  %v64 = phi i32 [ -2139062144, %b1 ], [ %v117, %b6 ]
  %v65 = phi <16 x i32> [ %v32, %b1 ], [ %v118, %b6 ]
  %v66 = phi <16 x i32> [ %v25, %b1 ], [ %v119, %b6 ]
  %v67 = phi <16 x i32> [ %v35, %b1 ], [ %v6, %b6 ]
  %v68 = icmp slt i32 %v58, %v63
  %v69 = select i1 %v68, i32 %v58, i32 %v63
  %v70 = icmp sgt i32 %v69, 0
  br i1 %v70, label %b3, label %b6

b3:                                               ; preds = %b2
  %v71 = xor i32 %v63, -1
  %v72 = icmp sgt i32 %v57, %v71
  %v73 = select i1 %v72, i32 %v57, i32 %v71
  %v74 = icmp sgt i32 %v73, -65
  %v75 = add i32 %v73, 63
  %v76 = select i1 %v74, i32 %v75, i32 -2
  %v77 = sub i32 %v76, %v73
  %v78 = lshr i32 %v77, 6
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v79 = phi i32 [ %v69, %b3 ], [ %v108, %b4 ]
  %v80 = phi ptr [ %v60, %b3 ], [ %v89, %b4 ]
  %v81 = phi ptr [ %v61, %b3 ], [ %v87, %b4 ]
  %v82 = phi ptr [ %v62, %b3 ], [ %v92, %b4 ]
  %v83 = phi i32 [ %v64, %b3 ], [ %v106, %b4 ]
  %v84 = phi <16 x i32> [ %v65, %b3 ], [ %v85, %b4 ]
  %v85 = phi <16 x i32> [ %v66, %b3 ], [ %v93, %b4 ]
  %v86 = phi <16 x i32> [ %v42, %b3 ], [ %v107, %b4 ]
  %v87 = getelementptr inbounds <16 x i32>, ptr %v81, i32 1
  %v88 = load <16 x i32>, ptr %v81, align 64, !tbaa !0
  %v89 = getelementptr inbounds <16 x i32>, ptr %v80, i32 1
  %v90 = load <16 x i32>, ptr %v80, align 64, !tbaa !0
  %v91 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v85, <16 x i32> %v84, i32 3)
  %v92 = getelementptr inbounds <16 x i32>, ptr %v82, i32 1
  %v93 = load <16 x i32>, ptr %v82, align 64, !tbaa !0
  %v94 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v93, <16 x i32> %v85, i32 3)
  %v95 = tail call <16 x i32> @llvm.hexagon.V6.vsububsat(<16 x i32> %v85, <16 x i32> %v21)
  %v96 = tail call <16 x i32> @llvm.hexagon.V6.vaddubsat(<16 x i32> %v85, <16 x i32> %v21)
  %v97 = tail call <16 x i32> @llvm.hexagon.V6.vmaxub(<16 x i32> %v88, <16 x i32> %v90)
  %v98 = tail call <16 x i32> @llvm.hexagon.V6.vminub(<16 x i32> %v88, <16 x i32> %v90)
  %v99 = tail call <16 x i32> @llvm.hexagon.V6.vmaxub(<16 x i32> %v94, <16 x i32> %v91)
  %v100 = tail call <16 x i32> @llvm.hexagon.V6.vminub(<16 x i32> %v94, <16 x i32> %v91)
  %v101 = tail call <16 x i32> @llvm.hexagon.V6.vminub(<16 x i32> %v97, <16 x i32> %v99)
  %v102 = tail call <16 x i32> @llvm.hexagon.V6.vmaxub(<16 x i32> %v98, <16 x i32> %v100)
  %v103 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v101, <16 x i32> %v96)
  %v104 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v95, <16 x i32> %v102)
  %v105 = tail call <64 x i1> @llvm.hexagon.V6.pred.or(<64 x i1> %v103, <64 x i1> %v104)
  %v106 = tail call i32 @llvm.hexagon.S6.rol.i.r(i32 %v83, i32 1)
  %v107 = tail call <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32> %v86, <64 x i1> %v105, i32 %v106)
  %v108 = add nsw i32 %v79, -64
  %v109 = icmp sgt i32 %v79, 64
  br i1 %v109, label %b4, label %b5

b5:                                               ; preds = %b4
  %v110 = add nuw nsw i32 %v78, 1
  %v111 = getelementptr <16 x i32>, ptr %v62, i32 %v110
  %v112 = getelementptr <16 x i32>, ptr %v60, i32 %v110
  %v113 = getelementptr <16 x i32>, ptr %v61, i32 %v110
  br label %b6

b6:                                               ; preds = %b5, %b2
  %v114 = phi ptr [ %v112, %b5 ], [ %v60, %b2 ]
  %v115 = phi ptr [ %v113, %b5 ], [ %v61, %b2 ]
  %v116 = phi ptr [ %v111, %b5 ], [ %v62, %b2 ]
  %v117 = phi i32 [ %v106, %b5 ], [ %v64, %b2 ]
  %v118 = phi <16 x i32> [ %v85, %b5 ], [ %v65, %b2 ]
  %v119 = phi <16 x i32> [ %v93, %b5 ], [ %v66, %b2 ]
  %v120 = phi <16 x i32> [ %v107, %b5 ], [ %v42, %b2 ]
  %v121 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> %v120, <16 x i32> %v67)
  %v122 = getelementptr inbounds <16 x i32>, ptr %v59, i32 1
  store <16 x i32> %v121, ptr %v59, align 64, !tbaa !0
  %v123 = add nsw i32 %v58, -512
  %v124 = icmp sgt i32 %v58, 512
  %v125 = add i32 %v57, 512
  br i1 %v124, label %b2, label %b7

b7:                                               ; preds = %b6
  br label %b8

b8:                                               ; preds = %b7, %b0
  %v127 = phi ptr [ %v56, %b7 ], [ %a4, %b0 ]
  %v128 = getelementptr inbounds <16 x i32>, ptr %v127, i32 -1
  %v129 = load <16 x i32>, ptr %v128, align 64, !tbaa !0
  %v130 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> %v129, <16 x i32> %v19)
  store <16 x i32> %v130, ptr %v128, align 64, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<64 x i1>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.pred.scalar2(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.vsplatrb(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32>, <64 x i1>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vd0() #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsububsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddubsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmaxub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vminub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.pred.or(<64 x i1>, <64 x i1>) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S6.rol.i.r(i32, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vand(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
