; RUN: llc -O3 -mtriple=hexagon < %s | FileCheck %s
; CHECK: v{{[0-9]+}} = vmem(r{{[0-9]+}}+#0)

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0) #0 {
b0:
  br i1 undef, label %b1, label %b5

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b4, %b1
  %v1 = phi i32 [ 0, %b1 ], [ %v50, %b4 ]
  %v2 = phi ptr [ %a0, %b1 ], [ undef, %b4 ]
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v3 = phi i32 [ -4, %b2 ], [ %v40, %b3 ]
  %v4 = add i32 0, -64
  %v5 = getelementptr inbounds i8, ptr null, i32 %v4
  %v7 = load <16 x i32>, ptr %v5, align 64, !tbaa !0
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> undef, <16 x i32> %v7, i32 4)
  %v9 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v8, <16 x i32> zeroinitializer)
  %v10 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v9, <16 x i32> undef)
  %v11 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v10, <16 x i32> undef, <16 x i32> undef)
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> zeroinitializer, <16 x i32> %v11, <16 x i32> undef)
  %v13 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> undef, <16 x i32> %v12, <16 x i32> undef)
  %v14 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> undef, <16 x i32> undef, i32 1)
  %v15 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %v14, <16 x i32> zeroinitializer)
  %v16 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> zeroinitializer, <16 x i32> zeroinitializer)
  %v17 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> zeroinitializer, <16 x i32> undef)
  %v18 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v15, <16 x i32> undef)
  %v19 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> zeroinitializer, <16 x i32> undef)
  %v20 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v16, <16 x i32> undef)
  %v21 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v19, <16 x i32> undef, <16 x i32> zeroinitializer)
  %v22 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v20, <16 x i32> undef, <16 x i32> zeroinitializer)
  %v23 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v22, <16 x i32> %v21)
  %v24 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> zeroinitializer, <32 x i32> %v23, i32 16843009)
  %v25 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v17, <16 x i32> %v13, <16 x i32> undef)
  %v26 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v18, <16 x i32> %v25, <16 x i32> undef)
  %v27 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v19, <16 x i32> %v26, <16 x i32> undef)
  %v28 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v20, <16 x i32> %v27, <16 x i32> undef)
  %v29 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> undef, <16 x i32> zeroinitializer)
  %v30 = tail call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> zeroinitializer, <16 x i32> zeroinitializer)
  %v31 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> undef, <16 x i32> undef)
  %v32 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v29, <16 x i32> undef)
  %v33 = tail call <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %v30, <16 x i32> undef)
  %v34 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v24, <32 x i32> zeroinitializer, i32 16843009)
  %v35 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %v34, <32 x i32> undef, i32 16843009)
  %v36 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> zeroinitializer, <16 x i32> %v28, <16 x i32> undef)
  %v37 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v31, <16 x i32> %v36, <16 x i32> undef)
  %v38 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v32, <16 x i32> %v37, <16 x i32> undef)
  %v39 = tail call <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1> %v33, <16 x i32> %v38, <16 x i32> undef)
  %v40 = add nsw i32 %v3, 3
  %v41 = icmp eq i32 %v40, 5
  br i1 %v41, label %b4, label %b3

b4:                                               ; preds = %b3
  %v42 = phi <16 x i32> [ %v39, %b3 ]
  %v43 = phi <32 x i32> [ %v35, %b3 ]
  %v44 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v43)
  %v45 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> zeroinitializer, <16 x i32> %v44, i32 -2)
  %v46 = tail call <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32> %v42)
  %v47 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v45)
  store <16 x i32> %v47, ptr %v2, align 64, !tbaa !0
  %v48 = getelementptr inbounds <16 x i32>, ptr null, i32 1
  %v49 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v46)
  store <16 x i32> %v49, ptr %v48, align 64, !tbaa !0
  %v50 = add nsw i32 %v1, 1
  %v51 = icmp slt i32 %v50, 0
  br i1 %v51, label %b2, label %b5

b5:                                               ; preds = %b4, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <64 x i1> @llvm.hexagon.V6.vgtub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddbnq(<64 x i1>, <16 x i32>, <16 x i32>) #1

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
