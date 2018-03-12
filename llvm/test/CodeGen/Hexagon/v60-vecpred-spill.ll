; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK-NOT: vmem(r30+#-1){{ *} = v{{[0-9]+}}
; CHECK-NOT: v{{[0-9]+}} = vmem(r30+#-1)
; CHECK: v{{[0-9]+}} = vmux

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(i8* nocapture readonly %a0, i32 %a1, i32 %a2, i32 %a3, i16* nocapture %a4, i16* nocapture %a5) #0 {
b0:
  %v0 = tail call i32 @llvm.hexagon.S2.vsplatrb(i32 %a3)
  %v1 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v0)
  %v2 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 16843009)
  %v3 = tail call <16 x i32> @llvm.hexagon.V6.vsubw(<16 x i32> undef, <16 x i32> undef)
  %v4 = sdiv i32 %a2, 64
  %v5 = icmp sgt i32 %a2, 63
  br i1 %v5, label %b1, label %b6

b1:                                               ; preds = %b0
  %v6 = bitcast i16* %a5 to <16 x i32>*
  %v7 = bitcast i16* %a4 to <16 x i32>*
  %v8 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v3, <16 x i32> %v3)
  br label %b2

b2:                                               ; preds = %b4, %b1
  %v9 = phi i32 [ 0, %b1 ], [ %v100, %b4 ]
  %v10 = phi i8* [ %a0, %b1 ], [ %v87, %b4 ]
  %v11 = phi <16 x i32>* [ %v6, %b1 ], [ %v99, %b4 ]
  %v12 = phi <16 x i32>* [ %v7, %b1 ], [ %v95, %b4 ]
  %v13 = bitcast i8* %v10 to <16 x i32>*
  %v14 = load <16 x i32>, <16 x i32>* %v13, align 64, !tbaa !0
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v15 = phi i32 [ -4, %b2 ], [ %v83, %b3 ]
  %v16 = phi <32 x i32> [ %v8, %b2 ], [ %v78, %b3 ]
  %v17 = phi <16 x i32> [ %v3, %b2 ], [ %v82, %b3 ]
  %v18 = mul nsw i32 %v15, %a1
  %v19 = getelementptr inbounds i8, i8* %v10, i32 %v18
  %v20 = bitcast i8* %v19 to <16 x i32>*
  %v21 = add i32 %v18, -64
  %v22 = getelementptr inbounds i8, i8* %v10, i32 %v21
  %v23 = bitcast i8* %v22 to <16 x i32>*
  %v24 = load <16 x i32>, <16 x i32>* %v23, align 64, !tbaa !0
  %v25 = load <16 x i32>, <16 x i32>* %v20, align 64, !tbaa !0
  %v26 = add i32 %v18, 64
  %v27 = getelementptr inbounds i8, i8* %v10, i32 %v26
  %v28 = bitcast i8* %v27 to <16 x i32>*
  %v29 = load <16 x i32>, <16 x i32>* %v28, align 64, !tbaa !0
  %v30 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v25, <16 x i32> %v14)
  %v31 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v30, <16 x i32> %v1)
  %v32 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v31, <16 x i32> %v3, <16 x i32> %v25)
  %v33 = tail call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %v16, <16 x i32> %v32, i32 16843009)
  %v34 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v31, <16 x i32> %v17, <16 x i32> %v2)
  %v35 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v25, <16 x i32> %v24, i32 1)
  %v36 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v29, <16 x i32> %v25, i32 1)
  %v37 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v25, <16 x i32> %v24, i32 2)
  %v38 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v29, <16 x i32> %v25, i32 2)
  %v39 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v35, <16 x i32> %v14)
  %v40 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v36, <16 x i32> %v14)
  %v41 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v37, <16 x i32> %v14)
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v38, <16 x i32> %v14)
  %v43 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v39, <16 x i32> %v1)
  %v44 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v40, <16 x i32> %v1)
  %v45 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v41, <16 x i32> %v1)
  %v46 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v42, <16 x i32> %v1)
  %v47 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v43, <16 x i32> %v3, <16 x i32> %v35)
  %v48 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v44, <16 x i32> %v3, <16 x i32> %v36)
  %v49 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v45, <16 x i32> %v3, <16 x i32> %v37)
  %v50 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v46, <16 x i32> %v3, <16 x i32> %v38)
  %v51 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v48, <16 x i32> %v47)
  %v52 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v33, <32 x i32> %v51, i32 16843009)
  %v53 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v50, <16 x i32> %v49)
  %v54 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v52, <32 x i32> %v53, i32 16843009)
  %v55 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v43, <16 x i32> %v34, <16 x i32> %v2)
  %v56 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v44, <16 x i32> %v55, <16 x i32> %v2)
  %v57 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v45, <16 x i32> %v56, <16 x i32> %v2)
  %v58 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v46, <16 x i32> %v57, <16 x i32> %v2)
  %v59 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v25, <16 x i32> %v24, i32 3)
  %v60 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v29, <16 x i32> %v25, i32 3)
  %v61 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v25, <16 x i32> %v24, i32 4)
  %v62 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v29, <16 x i32> %v25, i32 4)
  %v63 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v59, <16 x i32> %v14)
  %v64 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v60, <16 x i32> %v14)
  %v65 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v61, <16 x i32> %v14)
  %v66 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v62, <16 x i32> %v14)
  %v67 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v63, <16 x i32> %v1)
  %v68 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v64, <16 x i32> %v1)
  %v69 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v65, <16 x i32> %v1)
  %v70 = tail call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v66, <16 x i32> %v1)
  %v71 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v67, <16 x i32> %v3, <16 x i32> %v59)
  %v72 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v68, <16 x i32> %v3, <16 x i32> %v60)
  %v73 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v69, <16 x i32> %v3, <16 x i32> %v61)
  %v74 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %v70, <16 x i32> %v3, <16 x i32> %v62)
  %v75 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v72, <16 x i32> %v71)
  %v76 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v54, <32 x i32> %v75, i32 16843009)
  %v77 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v74, <16 x i32> %v73)
  %v78 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v76, <32 x i32> %v77, i32 16843009)
  %v79 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v67, <16 x i32> %v58, <16 x i32> %v2)
  %v80 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v68, <16 x i32> %v79, <16 x i32> %v2)
  %v81 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v69, <16 x i32> %v80, <16 x i32> %v2)
  %v82 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %v70, <16 x i32> %v81, <16 x i32> %v2)
  %v83 = add nsw i32 %v15, 1
  %v84 = icmp eq i32 %v83, 5
  br i1 %v84, label %b4, label %b3

b4:                                               ; preds = %b3
  %v85 = phi <16 x i32> [ %v82, %b3 ]
  %v86 = phi <32 x i32> [ %v78, %b3 ]
  %v87 = getelementptr inbounds i8, i8* %v10, i32 64
  %v88 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v86)
  %v89 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v86)
  %v90 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %v88, <16 x i32> %v89, i32 -2)
  %v91 = tail call <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32> %v85)
  %v92 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v90)
  %v93 = getelementptr inbounds <16 x i32>, <16 x i32>* %v12, i32 1
  store <16 x i32> %v92, <16 x i32>* %v12, align 64, !tbaa !0
  %v94 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v90)
  %v95 = getelementptr inbounds <16 x i32>, <16 x i32>* %v12, i32 2
  store <16 x i32> %v94, <16 x i32>* %v93, align 64, !tbaa !0
  %v96 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v91)
  %v97 = getelementptr inbounds <16 x i32>, <16 x i32>* %v11, i32 1
  store <16 x i32> %v96, <16 x i32>* %v11, align 64, !tbaa !0
  %v98 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v91)
  %v99 = getelementptr inbounds <16 x i32>, <16 x i32>* %v11, i32 2
  store <16 x i32> %v98, <16 x i32>* %v97, align 64, !tbaa !0
  %v100 = add nsw i32 %v9, 1
  %v101 = icmp slt i32 %v100, %v4
  br i1 %v101, label %b2, label %b5

b5:                                               ; preds = %b4
  br label %b6

b6:                                               ; preds = %b5, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.vsplatrb(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
