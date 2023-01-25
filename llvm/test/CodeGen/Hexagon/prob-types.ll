; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; REQUIRES: asserts

; This was aborting in Machine Loop Invariant Code Motion,
; we want to see something generated in assembly.
; CHECK: f0:

target triple = "hexagon"

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.128B(<32 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32>, <32 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32>, <32 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vlalignb.128B(<32 x i32>, <32 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32>, <32 x i32>) #0

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32>, <32 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrdelta.128B(<32 x i32>, <32 x i32>) #0

; Function Attrs: nounwind readnone
declare <64 x i32> @llvm.hexagon.V6.vunpackuh.128B(<32 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32>, <32 x i32>) #0

; Function Attrs: nounwind
define hidden void @f0(ptr %a0, ptr %a1, i32 %a2, <32 x i32> %a3, <32 x i32> %a4, <32 x i32> %a5, i32 %a6, <32 x i32> %a7) #1 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi ptr [ %v38, %b1 ], [ %a0, %b0 ]
  %v1 = phi ptr [ %v4, %b1 ], [ %a1, %b0 ]
  %v2 = phi i32 [ %v39, %b1 ], [ %a2, %b0 ]
  %v3 = phi <32 x i32> [ %v34, %b1 ], [ %a3, %b0 ]
  %v4 = getelementptr inbounds <32 x i32>, ptr %v1, i32 1
  %v5 = load <32 x i32>, ptr %v1, align 128, !tbaa !0
  %v6 = tail call <32 x i32> @llvm.hexagon.V6.vdmpybus.128B(<32 x i32> %v5, i32 16843009) #2
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> %v6, <32 x i32> %a4, i32 2) #2
  %v8 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v6, <32 x i32> %v7) #2
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> %v8, <32 x i32> %a4, i32 4) #2
  %v10 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v8, <32 x i32> %v9) #2
  %v11 = tail call <32 x i32> @llvm.hexagon.V6.vlalignb.128B(<32 x i32> %v10, <32 x i32> %a4, i32 8) #2
  %v12 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v10, <32 x i32> %v11) #2
  %v13 = tail call <32 x i32> @llvm.hexagon.V6.vlalignb.128B(<32 x i32> %v12, <32 x i32> %a4, i32 16) #2
  %v14 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v12, <32 x i32> %v13) #2
  %v15 = tail call <32 x i32> @llvm.hexagon.V6.vlalignb.128B(<32 x i32> %v14, <32 x i32> %a4, i32 32) #2
  %v16 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v14, <32 x i32> %v15) #2
  %v17 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v16, <32 x i32> %v15) #2
  %v18 = tail call <32 x i32> @llvm.hexagon.V6.vand.128B(<32 x i32> %v5, <32 x i32> %a5) #2
  %v19 = tail call <32 x i32> @llvm.hexagon.V6.vlalignbi.128B(<32 x i32> %v17, <32 x i32> %a4, i32 2) #2
  %v20 = tail call <32 x i32> @llvm.hexagon.V6.vaddh.128B(<32 x i32> %v19, <32 x i32> %v18) #2
  %v21 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %v17, <32 x i32> %v20, i32 -2) #2
  %v22 = tail call <32 x i32> @llvm.hexagon.V6.vrdelta.128B(<32 x i32> %v3, <32 x i32> %a7) #2
  %v23 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v21) #2
  %v24 = tail call <64 x i32> @llvm.hexagon.V6.vunpackuh.128B(<32 x i32> %v23) #2
  %v25 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v21) #2
  %v26 = tail call <64 x i32> @llvm.hexagon.V6.vunpackuh.128B(<32 x i32> %v25) #2
  %v27 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v24) #2
  %v28 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %v22, <32 x i32> %v27) #2
  %v29 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v24) #2
  %v30 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %v22, <32 x i32> %v29) #2
  %v31 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v26) #2
  %v32 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %v22, <32 x i32> %v31) #2
  %v33 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v26) #2
  %v34 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %v22, <32 x i32> %v33) #2
  %v35 = getelementptr inbounds <32 x i32>, ptr %v0, i32 1
  store <32 x i32> %v28, ptr %v0, align 128, !tbaa !0
  %v36 = getelementptr inbounds <32 x i32>, ptr %v0, i32 2
  store <32 x i32> %v30, ptr %v35, align 128, !tbaa !0
  %v37 = getelementptr inbounds <32 x i32>, ptr %v0, i32 3
  store <32 x i32> %v32, ptr %v36, align 128, !tbaa !0
  %v38 = getelementptr inbounds <32 x i32>, ptr %v0, i32 4
  store <32 x i32> %v34, ptr %v37, align 128, !tbaa !0
  %v39 = add nsw i32 %v2, 128
  %v40 = icmp slt i32 %v39, %a6
  br i1 %v40, label %b1, label %b2

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
