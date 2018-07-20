; RUN: llc -march=hexagon -O2 -disable-block-placement < %s | FileCheck %s
; Disable block placement because it intereferes with the generated code.

; CHECK:      if (p{{[0-9]*}}) jump:nt .LBB0_2
; CHECK-NEXT: v{{[0-9]*}} = vmem(r{{[0-9]*}}+#0)
; CHECK-NEXT: }

target triple = "hexagon"

@g0 = global <16 x i32> zeroinitializer, align 64

; Function Attrs: nounwind
define void @f0(i16* nocapture readonly %a0, i32 %a1, i32 %a2, i16* nocapture %a3) #0 {
b0:
  %v0 = mul i32 %a2, -2
  %v1 = add i32 %v0, 64
  %v2 = bitcast i16* %a3 to <16 x i32>*
  %v3 = load <16 x i32>, <16 x i32>* @g0, align 64
  %v4 = sdiv i32 %a1, 32
  %v5 = icmp sgt i32 %a1, 31
  br i1 %v5, label %b1, label %b4

b1:                                               ; preds = %b0
  %v6 = bitcast i16* %a0 to <16 x i32>*
  %v7 = icmp sgt i32 %a1, 63
  %v8 = mul i32 %v4, 32
  %v9 = select i1 %v7, i32 %v8, i32 32
  %v10 = getelementptr i16, i16* %a3, i32 %v9
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v11 = phi i32 [ 0, %b1 ], [ %v19, %b2 ]
  %v12 = phi <16 x i32> [ %v3, %b1 ], [ %v16, %b2 ]
  %v13 = phi <16 x i32>* [ %v2, %b1 ], [ %v18, %b2 ]
  %v14 = phi <16 x i32>* [ %v6, %b1 ], [ %v15, %b2 ]
  %v15 = getelementptr inbounds <16 x i32>, <16 x i32>* %v14, i32 1
  %v16 = load <16 x i32>, <16 x i32>* %v14, align 64
  %v17 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v16, <16 x i32> %v12, i32 %v1)
  %v18 = getelementptr inbounds <16 x i32>, <16 x i32>* %v13, i32 1
  store <16 x i32> %v17, <16 x i32>* %v13, align 64
  %v19 = add nsw i32 %v11, 1
  %v20 = icmp slt i32 %v19, %v4
  br i1 %v20, label %b2, label %b3

b3:                                               ; preds = %b2
  %v21 = bitcast i16* %v10 to <16 x i32>*
  %v22 = load <16 x i32>, <16 x i32>* @g0, align 64
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v23 = phi <16 x i32> [ %v22, %b3 ], [ %v3, %b0 ]
  %v24 = phi <16 x i32> [ %v16, %b3 ], [ %v3, %b0 ]
  %v25 = phi <16 x i32>* [ %v21, %b3 ], [ %v2, %b0 ]
  %v26 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v23, <16 x i32> %v24, i32 %v1)
  store <16 x i32> %v26, <16 x i32>* %v25, align 64
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv62" "target-features"="+hvxv62,+hvx-length64b" }
attributes #1 = { nounwind readnone }
