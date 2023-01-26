; RUN: llc -march=hexagon -O2 -simplifycfg-require-and-preserve-domtree=1 < %s
; REQUIRES: asserts

; Test that we generate the correct Phi names in the epilog when we need
; to reuse an existing Phi. This bug caused an assert in live variable
; analysis because the wrong virtual register was used.
; The bug occurs when a Phi references another Phi, and referent Phi
; value is used in multiple stages. When this occurs, the referring Phi
; can reuse one of the new values. We have code that deals with this in the
; kernel, but this case can occur in the epilog too.

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32>, <16 x i32>, i32) #0

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32>, <16 x i32>) #0

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, i32 %a1, i32 %a2) #1 {
b0:
  %v0 = mul nsw i32 %a1, 2
  br i1 undef, label %b1, label %b5

b1:                                               ; preds = %b0
  %v1 = getelementptr inbounds i8, ptr %a0, i32 %v0
  %v2 = icmp sgt i32 %a2, 64
  %v3 = add i32 %v0, 64
  %v4 = add i32 %a1, 64
  %v5 = sub i32 64, %a1
  %v6 = sub i32 64, %v0
  br i1 %v2, label %b2, label %b4

b2:                                               ; preds = %b1
  %v7 = getelementptr inbounds i8, ptr %v1, i32 %v3
  %v8 = getelementptr inbounds i8, ptr %v1, i32 %v4
  %v9 = getelementptr inbounds i8, ptr %v1, i32 64
  %v10 = getelementptr inbounds i8, ptr %v1, i32 %v5
  %v11 = getelementptr inbounds i8, ptr %v1, i32 %v6
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v17 = phi ptr [ null, %b2 ], [ %v52, %b3 ]
  %v18 = phi ptr [ %v7, %b2 ], [ %v34, %b3 ]
  %v19 = phi ptr [ %v8, %b2 ], [ %v32, %b3 ]
  %v20 = phi ptr [ %v9, %b2 ], [ %v30, %b3 ]
  %v21 = phi ptr [ %v10, %b2 ], [ %v28, %b3 ]
  %v22 = phi ptr [ %v11, %b2 ], [ %v26, %b3 ]
  %v23 = phi <32 x i32> [ undef, %b2 ], [ %v37, %b3 ]
  %v24 = phi <32 x i32> [ zeroinitializer, %b2 ], [ %v23, %b3 ]
  %v25 = phi i32 [ %a2, %b2 ], [ %v53, %b3 ]
  %v26 = getelementptr inbounds <16 x i32>, ptr %v22, i32 1
  %v27 = load <16 x i32>, ptr %v22, align 64
  %v28 = getelementptr inbounds <16 x i32>, ptr %v21, i32 1
  %v29 = load <16 x i32>, ptr %v21, align 64
  %v30 = getelementptr inbounds <16 x i32>, ptr %v20, i32 1
  %v31 = load <16 x i32>, ptr %v20, align 64
  %v32 = getelementptr inbounds <16 x i32>, ptr %v19, i32 1
  %v33 = load <16 x i32>, ptr %v19, align 64
  %v34 = getelementptr inbounds <16 x i32>, ptr %v18, i32 1
  %v35 = load <16 x i32>, ptr %v18, align 64
  %v36 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v33, <16 x i32> %v29) #3
  %v37 = tail call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> undef, <32 x i32> %v36, i32 67372036) #3
  %v38 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v23) #3
  %v39 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v24) #3
  %v40 = tail call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %v38, <16 x i32> %v39, i32 2) #3
  %v41 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v37) #3
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v41, <16 x i32> %v38, i32 2) #3
  %v43 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v37) #3
  %v44 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %v43, <16 x i32> undef, i32 2) #3
  %v45 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v40, <16 x i32> %v42) #3
  %v46 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %v45, <16 x i32> %v38, i32 101058054) #3
  %v47 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %v46, <16 x i32> zeroinitializer, i32 67372036) #3
  %v48 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> undef, <16 x i32> %v44) #3
  %v49 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %v48, <16 x i32> undef, i32 101058054) #3
  %v50 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %v49, <16 x i32> zeroinitializer, i32 67372036) #3
  %v51 = tail call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> %v50, <16 x i32> %v47) #3
  %v52 = getelementptr inbounds <16 x i32>, ptr %v17, i32 1
  store <16 x i32> %v51, ptr %v17, align 64
  %v53 = add nsw i32 %v25, -64
  %v54 = icmp sgt i32 %v53, 64
  br i1 %v54, label %b3, label %b4

b4:                                               ; preds = %b3, %b1
  unreachable

b5:                                               ; preds = %b0
  ret void
}

; Function Attrs: nounwind
define void @f1(i32 %a0, ptr %a1) #1 {
b0:
  %v0 = ptrtoint ptr %a1 to i32
  %v1 = ashr i32 %a0, 1
  %v2 = tail call i32 @llvm.hexagon.A2.combine.ll(i32 undef, i32 undef)
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b1, label %b2

b2:                                               ; preds = %b2, %b1, %b0
  %v3 = phi i64 [ %v11, %b2 ], [ undef, %b0 ], [ undef, %b1 ]
  %v4 = phi i32 [ %v12, %b2 ], [ 0, %b0 ], [ undef, %b1 ]
  %v5 = phi i32 [ %v6, %b2 ], [ %v2, %b0 ], [ undef, %b1 ]
  %v6 = phi i32 [ %v10, %b2 ], [ undef, %b0 ], [ undef, %b1 ]
  %v7 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 undef, i64 %v3, i64 undef)
  %v8 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v5, i32 %v5)
  %v9 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 undef, i64 %v8, i64 undef)
  %v10 = tail call i32 @llvm.hexagon.A2.combine.ll(i32 0, i32 undef)
  %v11 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v10, i32 %v6)
  %v12 = add nsw i32 %v4, 1
  %v13 = icmp eq i32 %v12, %v1
  br i1 %v13, label %b3, label %b2

b3:                                               ; preds = %b2
  %v14 = phi i64 [ %v9, %b2 ]
  %v15 = phi i64 [ %v7, %b2 ]
  %v16 = trunc i64 %v14 to i32
  %v17 = trunc i64 %v15 to i32
  %v18 = inttoptr i32 %v0 to ptr
  store i32 %v17, ptr %v18, align 4
  store i32 %v16, ptr undef, align 4
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.combine.ll(i32, i32) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #0

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #2 = { noreturn nounwind }
attributes #3 = { nounwind }
