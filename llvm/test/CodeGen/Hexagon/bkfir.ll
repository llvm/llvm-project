; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts
; Check for successful compilation.

target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind optsize
define void @f0(ptr nocapture readonly %a0, ptr nocapture readonly %a1, ptr nocapture %a2, i32 %a3, i32 %a4, i32 %a5) #0 {
b0:
  %v2 = icmp sgt i32 %a5, 0
  br i1 %v2, label %b1, label %b6

b1:                                               ; preds = %b0
  %v3 = icmp sgt i32 %a4, 0
  %v4 = getelementptr i16, ptr %a2, i32 5
  %v5 = getelementptr i16, ptr %a2, i32 6
  %v6 = getelementptr i16, ptr %a2, i32 7
  br label %b2

b2:                                               ; preds = %b5, %b1
  %v7 = phi ptr [ %a2, %b1 ], [ %v12, %b5 ]
  %v8 = phi ptr [ %v4, %b1 ], [ %v59, %b5 ]
  %v9 = phi ptr [ %v5, %b1 ], [ %v60, %b5 ]
  %v10 = phi ptr [ %v6, %b1 ], [ %v61, %b5 ]
  %v11 = phi i32 [ 0, %b1 ], [ %v57, %b5 ]
  %v12 = getelementptr i16, ptr %v7, i32 4
  br i1 %v3, label %b3, label %b5

b3:                                               ; preds = %b3, %b2
  %v13 = phi i32 [ %v43, %b3 ], [ 0, %b2 ]
  %v14 = phi i32 [ %v30, %b3 ], [ 0, %b2 ]
  %v15 = phi i32 [ %v34, %b3 ], [ 0, %b2 ]
  %v16 = phi i32 [ %v42, %b3 ], [ 0, %b2 ]
  %v17 = phi i32 [ %v38, %b3 ], [ 0, %b2 ]
  %v18 = add nsw i32 %v13, %v11
  %v19 = sdiv i32 %v18, 4
  %v20 = getelementptr inbounds i64, ptr %a0, i32 %v19
  %v21 = load i64, ptr %v20, align 8
  %v22 = add nsw i32 %v19, 1
  %v23 = getelementptr inbounds i64, ptr %a0, i32 %v22
  %v24 = load i64, ptr %v23, align 8
  %v25 = sdiv i32 %v13, 4
  %v26 = getelementptr inbounds i64, ptr %a1, i32 %v25
  %v27 = load i64, ptr %v26, align 8
  %v28 = sext i32 %v14 to i64
  %v29 = tail call i64 @llvm.hexagon.M2.vrmac.s0(i64 %v28, i64 %v21, i64 %v27)
  %v30 = trunc i64 %v29 to i32
  %v31 = tail call i64 @llvm.hexagon.S2.valignib(i64 %v24, i64 %v21, i32 2)
  %v32 = sext i32 %v15 to i64
  %v33 = tail call i64 @llvm.hexagon.M2.vrmac.s0(i64 %v32, i64 %v31, i64 %v27)
  %v34 = trunc i64 %v33 to i32
  %v35 = tail call i64 @llvm.hexagon.S2.valignib(i64 %v24, i64 %v21, i32 4)
  %v36 = sext i32 %v17 to i64
  %v37 = tail call i64 @llvm.hexagon.M2.vrmac.s0(i64 %v36, i64 %v35, i64 %v27)
  %v38 = trunc i64 %v37 to i32
  %v39 = tail call i64 @llvm.hexagon.S2.valignib(i64 %v24, i64 %v21, i32 6)
  %v40 = sext i32 %v16 to i64
  %v41 = tail call i64 @llvm.hexagon.M2.vrmac.s0(i64 %v40, i64 %v39, i64 %v27)
  %v42 = trunc i64 %v41 to i32
  %v43 = add nsw i32 %v13, 4
  %v44 = icmp slt i32 %v43, %a4
  br i1 %v44, label %b3, label %b4

b4:                                               ; preds = %b3
  %v45 = ashr i32 %v30, 18
  %v46 = trunc i32 %v45 to i16
  %v47 = ashr i32 %v34, 18
  %v48 = trunc i32 %v47 to i16
  %v49 = ashr i32 %v38, 18
  %v50 = trunc i32 %v49 to i16
  %v51 = ashr i32 %v42, 18
  %v52 = trunc i32 %v51 to i16
  br label %b5

b5:                                               ; preds = %b4, %b2
  %v53 = phi i16 [ %v46, %b4 ], [ 0, %b2 ]
  %v54 = phi i16 [ %v48, %b4 ], [ 0, %b2 ]
  %v55 = phi i16 [ %v52, %b4 ], [ 0, %b2 ]
  %v56 = phi i16 [ %v50, %b4 ], [ 0, %b2 ]
  %v57 = add nsw i32 %v11, 4
  store i16 %v53, ptr %v12, align 8
  store i16 %v54, ptr %v8, align 8
  store i16 %v56, ptr %v9, align 8
  store i16 %v55, ptr %v10, align 8
  %v58 = icmp slt i32 %v57, %a5
  %v59 = getelementptr i16, ptr %v8, i32 4
  %v60 = getelementptr i16, ptr %v9, i32 4
  %v61 = getelementptr i16, ptr %v10, i32 4
  br i1 %v58, label %b2, label %b6

b6:                                               ; preds = %b5, %b0
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vrmac.s0(i64, i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.valignib(i64, i64, i32) #1

attributes #0 = { nounwind optsize }
attributes #1 = { nounwind readnone }
