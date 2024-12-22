; RUN: llc -march=hexagon -enable-pipeliner -debug-only=pipeliner < %s -o - 2>&1 > /dev/null -pipeliner-experimental-cg=true | FileCheck %s
; REQUIRES: asserts

; Test that checks that we compute the correct ResMII for haar.

; CHECK: MII = 4 MAX_II = 14 (rec=1, res=4)

; Function Attrs: nounwind
define void @f0(ptr noalias nocapture readonly %a0, i32 %a1, i32 %a2, i32 %a3, ptr noalias nocapture %a4, i32 %a5) #0 {
b0:
  %v0 = ashr i32 %a3, 2
  %v1 = ashr i32 %a3, 1
  %v2 = add i32 %v1, %v0
  %v3 = icmp sgt i32 %a2, 0
  br i1 %v3, label %b1, label %b8

b1:                                               ; preds = %b0
  %v4 = sdiv i32 %a1, 64
  %v5 = icmp sgt i32 %a1, 63
  br label %b2

b2:                                               ; preds = %b6, %b1
  %v6 = phi i32 [ 0, %b1 ], [ %v56, %b6 ]
  %v7 = ashr exact i32 %v6, 1
  %v8 = mul nsw i32 %v7, %a3
  br i1 %v5, label %b3, label %b6

b3:                                               ; preds = %b2
  %v9 = add nsw i32 %v6, 1
  %v10 = mul nsw i32 %v9, %a5
  %v11 = mul nsw i32 %v6, %a5
  %v12 = add i32 %v2, %v8
  %v13 = add i32 %v8, %v0
  %v14 = add i32 %v8, %v1
  %v15 = getelementptr inbounds i8, ptr %a4, i32 %v10
  %v16 = getelementptr inbounds i8, ptr %a4, i32 %v11
  %v17 = getelementptr inbounds i16, ptr %a0, i32 %v12
  %v18 = getelementptr inbounds i16, ptr %a0, i32 %v13
  %v19 = getelementptr inbounds i16, ptr %a0, i32 %v14
  %v20 = getelementptr inbounds i16, ptr %a0, i32 %v8
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v27 = phi i32 [ 0, %b3 ], [ %v54, %b4 ]
  %v28 = phi ptr [ %v20, %b3 ], [ %v34, %b4 ]
  %v29 = phi ptr [ %v19, %b3 ], [ %v36, %b4 ]
  %v30 = phi ptr [ %v18, %b3 ], [ %v38, %b4 ]
  %v31 = phi ptr [ %v17, %b3 ], [ %v40, %b4 ]
  %v32 = phi ptr [ %v15, %b3 ], [ %v53, %b4 ]
  %v33 = phi ptr [ %v16, %b3 ], [ %v52, %b4 ]
  %v34 = getelementptr inbounds <16 x i32>, ptr %v28, i32 1
  %v35 = load <16 x i32>, ptr %v28, align 64
  %v36 = getelementptr inbounds <16 x i32>, ptr %v29, i32 1
  %v37 = load <16 x i32>, ptr %v29, align 64
  %v38 = getelementptr inbounds <16 x i32>, ptr %v30, i32 1
  %v39 = load <16 x i32>, ptr %v30, align 64
  %v40 = getelementptr inbounds <16 x i32>, ptr %v31, i32 1
  %v41 = load <16 x i32>, ptr %v31, align 64
  %v42 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v35, <16 x i32> %v37)
  %v43 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v35, <16 x i32> %v37)
  %v44 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %v39, <16 x i32> %v41)
  %v45 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %v39, <16 x i32> %v41)
  %v46 = tail call <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32> %v42, <16 x i32> %v44)
  %v47 = tail call <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32> %v42, <16 x i32> %v44)
  %v48 = tail call <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32> %v43, <16 x i32> %v45)
  %v49 = tail call <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32> %v43, <16 x i32> %v45)
  %v50 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v47, <16 x i32> %v46)
  %v51 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %v49, <16 x i32> %v48)
  %v52 = getelementptr inbounds <16 x i32>, ptr %v33, i32 1
  store <16 x i32> %v50, ptr %v33, align 64
  %v53 = getelementptr inbounds <16 x i32>, ptr %v32, i32 1
  store <16 x i32> %v51, ptr %v32, align 64
  %v54 = add nsw i32 %v27, 1
  %v55 = icmp slt i32 %v54, %v4
  br i1 %v55, label %b4, label %b5

b5:                                               ; preds = %b4
  br label %b6

b6:                                               ; preds = %b5, %b2
  %v56 = add nsw i32 %v6, 2
  %v57 = icmp slt i32 %v56, %a2
  br i1 %v57, label %b2, label %b7

b7:                                               ; preds = %b6
  br label %b8

b8:                                               ; preds = %b7, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
