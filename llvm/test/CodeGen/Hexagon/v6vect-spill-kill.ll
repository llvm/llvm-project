; RUN: llc -march=hexagon -O3 < %s
; REQUIRES: asserts

; Test that we don't assert because of requiring too many scavenger spill
; slots. This happens because the kill flag wasn't added to the appropriate
; operands for the spill code.

define void @f0(i32 %a0, i8* noalias nocapture %a1) #0 {
b0:
  %v0 = tail call <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32> undef)
  %v1 = sdiv i32 %a0, 128
  %v2 = icmp sgt i32 %a0, 127
  br i1 %v2, label %b1, label %b3

b1:                                               ; preds = %b0
  %v3 = bitcast i8* %a1 to <32 x i32>*
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v4 = phi <32 x i32>* [ %v3, %b1 ], [ undef, %b2 ]
  %v5 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> undef, <32 x i32> zeroinitializer, i32 2)
  %v6 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v5, <32 x i32> zeroinitializer)
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> zeroinitializer, <32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %v8 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v7, <32 x i32> zeroinitializer)
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> zeroinitializer, <32 x i32> %v8, <32 x i32> zeroinitializer)
  %v10 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v9, <32 x i32> zeroinitializer)
  %v11 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> undef, <32 x i32> zeroinitializer, i32 4)
  %v12 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v11, <32 x i32> zeroinitializer)
  %v13 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer)
  %v14 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> zeroinitializer, <32 x i32> undef)
  %v15 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v12, <32 x i32> undef)
  %v16 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v13, <32 x i32> undef)
  %v17 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v10, <32 x i32> zeroinitializer)
  %v18 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v14, <32 x i32> %v17, <32 x i32> zeroinitializer)
  %v19 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v15, <32 x i32> %v18, <32 x i32> zeroinitializer)
  %v20 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v16, <32 x i32> %v19, <32 x i32> zeroinitializer)
  %v21 = getelementptr inbounds i8, i8* null, i32 undef
  %v22 = bitcast i8* %v21 to <32 x i32>*
  %v23 = load <32 x i32>, <32 x i32>* %v22, align 128, !tbaa !0
  %v24 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v23, <32 x i32> zeroinitializer)
  %v25 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v24, <32 x i32> undef)
  %v26 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v25, <32 x i32> %v20, <32 x i32> zeroinitializer)
  %v27 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v26, <32 x i32> zeroinitializer)
  %v28 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v27, <32 x i32> zeroinitializer)
  %v29 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v28, <32 x i32> zeroinitializer)
  %v30 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v29, <32 x i32> zeroinitializer)
  %v31 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v30, <32 x i32> zeroinitializer)
  %v32 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v31, <32 x i32> zeroinitializer)
  %v33 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v32, <32 x i32> zeroinitializer)
  %v34 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v33, <32 x i32> zeroinitializer)
  %v35 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v34, <32 x i32> zeroinitializer)
  %v36 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> undef, <32 x i32> undef, i32 1)
  %v37 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> undef, <32 x i32> undef, i32 1)
  %v38 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> undef, <32 x i32> undef, i32 2)
  %v39 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v36, <32 x i32> zeroinitializer)
  %v40 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v37, <32 x i32> zeroinitializer)
  %v41 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v38, <32 x i32> zeroinitializer)
  %v42 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v39, <32 x i32> undef)
  %v43 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v40, <32 x i32> undef)
  %v44 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v41, <32 x i32> undef)
  %v45 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> undef, <32 x i32> undef)
  %v46 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v42, <32 x i32> %v35, <32 x i32> zeroinitializer)
  %v47 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v43, <32 x i32> %v46, <32 x i32> zeroinitializer)
  %v48 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v44, <32 x i32> %v47, <32 x i32> zeroinitializer)
  %v49 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v45, <32 x i32> %v48, <32 x i32> zeroinitializer)
  %v50 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> undef, <32 x i32> undef, i32 4)
  %v51 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> undef, <32 x i32> undef, i32 4)
  %v52 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> undef, <32 x i32> zeroinitializer)
  %v53 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v50, <32 x i32> zeroinitializer)
  %v54 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v51, <32 x i32> zeroinitializer)
  %v55 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v52, <32 x i32> undef)
  %v56 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v53, <32 x i32> undef)
  %v57 = tail call <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v54, <32 x i32> undef)
  %v58 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> undef, <32 x i32> %v49, <32 x i32> zeroinitializer)
  %v59 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v55, <32 x i32> %v58, <32 x i32> zeroinitializer)
  %v60 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v56, <32 x i32> %v59, <32 x i32> zeroinitializer)
  %v61 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1> %v57, <32 x i32> %v60, <32 x i32> zeroinitializer)
  %v62 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> zeroinitializer, <32 x i32> %v61, <32 x i32> undef, i32 5)
  %v63 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuhv.128B(<32 x i32> undef, <32 x i32> undef)
  %v64 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v62)
  %v65 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuhv.128B(<32 x i32> zeroinitializer, <32 x i32> %v64)
  %v66 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v63)
  %v67 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v63)
  %v68 = tail call <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32> %v66, <32 x i32> %v67, i32 14)
  %v69 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v65)
  %v70 = tail call <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32> %v69, <32 x i32> undef, i32 14)
  %v71 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> %v70, <32 x i32> %v68)
  store <32 x i32> %v71, <32 x i32>* %v4, align 128, !tbaa !0
  %v72 = icmp slt i32 0, %v1
  br i1 %v72, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <1024 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<1024 x i1>, <32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32>, <32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vmpyuhv.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32>, <32 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
