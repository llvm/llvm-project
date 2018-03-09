; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; CHECK: v{{[0-9]+}}.h{{ *}}={{ *}}vadd(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
; CHECK: }
; CHECK: {
; CHECK: v{{[0-9]+}}{{ *}}={{ *}}valign(v{{[0-9]+}},v{{[0-9]+}},r{{[0-9]+}})
; CHECK: }
; CHECK: {
; CHECK: v{{[0-9]+}}{{ *}}={{ *}}valign(v{{[0-9]+}},v{{[0-9]+}},r{{[0-9]+}})

target triple = "hexagon"

@ZERO = global <16 x i32> zeroinitializer, align 64

define void @fred(i16* nocapture readonly %a0, i32 %a1, i32 %a2, i16* nocapture %a3) #0 {
b4:
  %v5 = bitcast i16* %a0 to <16 x i32>*
  %v6 = getelementptr inbounds i16, i16* %a0, i32 %a1
  %v7 = bitcast i16* %v6 to <16 x i32>*
  %v8 = mul nsw i32 %a1, 2
  %v9 = getelementptr inbounds i16, i16* %a0, i32 %v8
  %v10 = bitcast i16* %v9 to <16 x i32>*
  %v11 = load <16 x i32>, <16 x i32>* %v5, align 64, !tbaa !1
  %v12 = load <16 x i32>, <16 x i32>* %v7, align 64, !tbaa !1
  %v13 = load <16 x i32>, <16 x i32>* %v10, align 64, !tbaa !1
  %v14 = load <16 x i32>, <16 x i32>* @ZERO, align 64, !tbaa !1
  %v15 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v14, <16 x i32> %v14)
  %v16 = sdiv i32 %a2, 32
  %v17 = icmp sgt i32 %a2, 31
  br i1 %v17, label %b18, label %b66

b18:                                              ; preds = %b4
  %v19 = add i32 %v8, 32
  %v20 = add i32 %a1, 32
  %v21 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v12, <16 x i32> %v12)
  %v22 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v11, <16 x i32> %v13)
  %v23 = getelementptr inbounds i16, i16* %a0, i32 %v19
  %v24 = getelementptr inbounds i16, i16* %a0, i32 %v20
  %v25 = getelementptr inbounds i16, i16* %a0, i32 32
  %v26 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v11, <16 x i32> %v13)
  %v27 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v22, <16 x i32> %v21)
  %v28 = bitcast i16* %v23 to <16 x i32>*
  %v29 = bitcast i16* %v24 to <16 x i32>*
  %v30 = bitcast i16* %v25 to <16 x i32>*
  %v31 = bitcast i16* %a3 to <16 x i32>*
  br label %b32

b32:                                              ; preds = %b32, %b18
  %v33 = phi i32 [ 0, %b18 ], [ %v63, %b32 ]
  %v34 = phi <16 x i32>* [ %v31, %b18 ], [ %v62, %b32 ]
  %v35 = phi <16 x i32>* [ %v28, %b18 ], [ %v46, %b32 ]
  %v36 = phi <16 x i32>* [ %v29, %b18 ], [ %v44, %b32 ]
  %v37 = phi <16 x i32>* [ %v30, %b18 ], [ %v42, %b32 ]
  %v38 = phi <16 x i32> [ %v15, %b18 ], [ %v39, %b32 ]
  %v39 = phi <16 x i32> [ %v26, %b18 ], [ %v56, %b32 ]
  %v40 = phi <16 x i32> [ %v27, %b18 ], [ %v51, %b32 ]
  %v41 = phi <16 x i32> [ %v15, %b18 ], [ %v40, %b32 ]
  %v42 = getelementptr inbounds <16 x i32>, <16 x i32>* %v37, i32 1
  %v43 = load <16 x i32>, <16 x i32>* %v37, align 64, !tbaa !1
  %v44 = getelementptr inbounds <16 x i32>, <16 x i32>* %v36, i32 1
  %v45 = load <16 x i32>, <16 x i32>* %v36, align 64, !tbaa !1
  %v46 = getelementptr inbounds <16 x i32>, <16 x i32>* %v35, i32 1
  %v47 = load <16 x i32>, <16 x i32>* %v35, align 64, !tbaa !1
  %v48 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v43, <16 x i32> %v47)
  %v49 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v45, <16 x i32> %v45)
  %v50 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v40, <16 x i32> %v41, i32 62)
  %v51 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v48, <16 x i32> %v49)
  %v52 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v51, <16 x i32> %v40, i32 2)
  %v53 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32> %v50, <16 x i32> %v52)
  %v54 = getelementptr inbounds <16 x i32>, <16 x i32>* %v34, i32 1
  store <16 x i32> %v53, <16 x i32>* %v34, align 64, !tbaa !1
  %v55 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v39, <16 x i32> %v38, i32 62)
  %v56 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v43, <16 x i32> %v47)
  %v57 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v56, <16 x i32> %v39, i32 2)
  %v58 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v39, <16 x i32> %v39)
  %v59 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v58, <16 x i32> %v55)
  %v60 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v59, <16 x i32> %v57)
  %v61 = tail call <16 x i32> @llvm.hexagon.V6.vabsh(<16 x i32> %v60)
  %v62 = getelementptr inbounds <16 x i32>, <16 x i32>* %v34, i32 2
  store <16 x i32> %v61, <16 x i32>* %v54, align 64, !tbaa !1
  %v63 = add nsw i32 %v33, 1
  %v64 = icmp slt i32 %v63, %v16
  br i1 %v64, label %b32, label %b65

b65:                                              ; preds = %b32
  br label %b66

b66:                                              ; preds = %b65, %b4
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32>, <16 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32>, <16 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.vabsh(<16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
