; RUN: llc -march=hexagon -O3 < %s
; REQUIRES: asserts

; Test that we don't assert because the compiler generates the wrong register
; class for the vector spill code in 128B mode.

define void @f0(i32 %a0) #0 {
b0:
  %v0 = tail call <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32 16843009)
  %v1 = tail call <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32> undef)
  %v2 = tail call <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32> zeroinitializer)
  %v3 = sdiv i32 %a0, 128
  %v4 = icmp sgt i32 %a0, 127
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v5 = phi i32 [ %v77, %b1 ], [ 0, %b0 ]
  %v6 = phi ptr [ undef, %b1 ], [ undef, %b0 ]
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> undef, <32 x i32> undef)
  %v8 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v7, <32 x i32> zeroinitializer)
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v8, <32 x i32> undef, <32 x i32> %v0)
  %v10 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> undef, <32 x i32> undef, i32 3)
  %v11 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> zeroinitializer, <32 x i32> undef)
  %v12 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v10, <32 x i32> undef)
  %v13 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v11, <32 x i32> zeroinitializer)
  %v14 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v12, <32 x i32> zeroinitializer)
  %v15 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v13, <32 x i32> %v9, <32 x i32> %v0)
  %v16 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v14, <32 x i32> %v15, <32 x i32> %v0)
  %v17 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v16, <32 x i32> %v0)
  %v18 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v17, <32 x i32> %v0)
  %v19 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> undef, <32 x i32> zeroinitializer)
  %v20 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v19, <32 x i32> %v18, <32 x i32> %v0)
  %v21 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> undef, <32 x i32> zeroinitializer)
  %v22 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> undef, <32 x i32> undef, <32 x i32> undef)
  %v23 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v21, <32 x i32> undef, <32 x i32> undef)
  %v24 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v23, <32 x i32> %v22)
  %v25 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> zeroinitializer, <64 x i32> %v24, i32 16843009)
  %v26 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v20, <32 x i32> %v0)
  %v27 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v26, <32 x i32> %v0)
  %v28 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v27, <32 x i32> %v0)
  %v29 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v21, <32 x i32> %v28, <32 x i32> %v0)
  %v30 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> undef, <32 x i32> zeroinitializer)
  %v31 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> undef, <32 x i32> undef, <32 x i32> zeroinitializer)
  %v32 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v31, <32 x i32> undef)
  %v33 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v25, <64 x i32> %v32, i32 16843009)
  %v34 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v33, <64 x i32> undef, i32 16843009)
  %v35 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v29, <32 x i32> %v0)
  %v36 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v35, <32 x i32> %v0)
  %v37 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v36, <32 x i32> %v0)
  %v38 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v30, <32 x i32> %v37, <32 x i32> %v0)
  %v39 = load <32 x i32>, ptr null, align 128, !tbaa !0
  %v40 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> undef, <32 x i32> zeroinitializer)
  %v41 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v40, <32 x i32> undef, <32 x i32> %v39)
  %v42 = tail call <64 x i32> @llvm.hexagon.V6.vmpybus.acc.128B(<64 x i32> %v34, <32 x i32> %v41, i32 16843009)
  %v43 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v40, <32 x i32> %v38, <32 x i32> %v0)
  %v44 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> %v39, <32 x i32> undef, i32 1)
  %v45 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> undef, <32 x i32> %v39, i32 1)
  %v46 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> undef, <32 x i32> %v39, i32 2)
  %v47 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v44, <32 x i32> undef)
  %v48 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v45, <32 x i32> undef)
  %v49 = tail call <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32> %v46, <32 x i32> undef)
  %v50 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v47, <32 x i32> zeroinitializer)
  %v51 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v48, <32 x i32> zeroinitializer)
  %v52 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> %v49, <32 x i32> zeroinitializer)
  %v53 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> %v52, <32 x i32> undef, <32 x i32> %v46)
  %v54 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v42, <64 x i32> undef, i32 16843009)
  %v55 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> %v53, <32 x i32> undef)
  %v56 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v54, <64 x i32> %v55, i32 16843009)
  %v57 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v50, <32 x i32> %v43, <32 x i32> %v0)
  %v58 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v51, <32 x i32> %v57, <32 x i32> %v0)
  %v59 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v58, <32 x i32> %v0)
  %v60 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v52, <32 x i32> %v59, <32 x i32> %v0)
  %v61 = tail call <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32> undef, <32 x i32> zeroinitializer)
  %v62 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v56, <64 x i32> undef, i32 16843009)
  %v63 = tail call <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32> %v62, <64 x i32> zeroinitializer, i32 16843009)
  %v64 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v60, <32 x i32> %v0)
  %v65 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> %v61, <32 x i32> %v64, <32 x i32> %v0)
  %v66 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v65, <32 x i32> %v0)
  %v67 = tail call <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1> undef, <32 x i32> %v66, <32 x i32> %v0)
  %v68 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> undef, <32 x i32> %v67, <32 x i32> %v1, i32 3)
  %v69 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %v68, <32 x i32> %v67, <32 x i32> %v2, i32 4)
  %v70 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %v69, <32 x i32> %v67, <32 x i32> %v2, i32 5)
  %v71 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v63)
  %v72 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v70)
  %v73 = tail call <64 x i32> @llvm.hexagon.V6.vmpyuhv.128B(<32 x i32> %v71, <32 x i32> %v72)
  %v74 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v73)
  %v75 = tail call <32 x i32> @llvm.hexagon.V6.vasrwh.128B(<32 x i32> %v74, <32 x i32> undef, i32 14)
  %v76 = tail call <32 x i32> @llvm.hexagon.V6.vshuffeb.128B(<32 x i32> %v75, <32 x i32> undef)
  store <32 x i32> %v76, ptr %v6, align 128, !tbaa !0
  %v77 = add nsw i32 %v5, 1
  %v78 = icmp slt i32 %v77, %v3
  br i1 %v78, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.lvsplatw.128B(i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshuffh.128B(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vabsdiffub.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <128 x i1> @llvm.hexagon.V6.vgtub.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vmpybus.acc.128B(<64 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddbnq.128B(<128 x i1>, <32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vmpabus.acc.128B(<64 x i32>, <64 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32>, <32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vmpyuhv.128B(<32 x i32>, <32 x i32>) #1

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
