; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that the pipeliner correctly fixes up the pipelined CFG when the loop
; has a constant trip count, and the trip count is less than the number of
; prolog blocks. Prior to the bug, the pipeliner deleted one extra prolog and
; epilog stage. We check this by counting the number of sxth instructions.

; CHECK: r{{[0-9]+}} = sxth(r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = sxth(r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = sxth(r{{[0-9]+}})
; CHECK: r{{[0-9]+}} = sxth(r{{[0-9]+}})

; Function Attrs: nounwind readonly
define signext i16 @f0(i16* nocapture readonly %a0, i16* nocapture readnone %a1, i16* nocapture readonly %a2, i16* nocapture readonly %a3, i16 signext %a4, i16 signext %a5, i16 signext %a6) #0 {
b0:
  %v0 = icmp sgt i16 %a5, 0
  br i1 %v0, label %b1, label %b7

b1:                                               ; preds = %b0
  %v1 = load i16, i16* %a0, align 2
  %v2 = sext i16 %v1 to i32
  %v3 = load i16, i16* %a3, align 2
  %v4 = sext i16 %v3 to i32
  br label %b2

b2:                                               ; preds = %b6, %b1
  %v5 = phi i32 [ 2147483647, %b1 ], [ %v44, %b6 ]
  %v6 = phi i16 [ 0, %b1 ], [ %v45, %b6 ]
  %v7 = phi i16 [ 0, %b1 ], [ %v43, %b6 ]
  %v8 = phi i16* [ %a2, %b1 ], [ %v38, %b6 ]
  %v9 = load i16, i16* %v8, align 2
  %v10 = sext i16 %v9 to i32
  %v11 = tail call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %v2, i32 %v10)
  %v12 = shl i32 %v11, 16
  %v13 = ashr exact i32 %v12, 16
  %v14 = tail call i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32 %v13, i32 %v4)
  %v15 = tail call i32 @llvm.hexagon.M2.hmmpyl.s1(i32 %v14, i32 %v13)
  %v16 = tail call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v15, i32 10)
  %v17 = getelementptr inbounds i16, i16* %v8, i32 1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v18 = phi i16* [ %v8, %b2 ], [ %v19, %b3 ]
  %v19 = phi i16* [ %v17, %b2 ], [ %v38, %b3 ]
  %v20 = phi i32 [ %v16, %b2 ], [ %v36, %b3 ]
  %v21 = phi i32 [ 1, %b2 ], [ %v37, %b3 ]
  %v22 = getelementptr inbounds i16, i16* %a0, i32 %v21
  %v23 = load i16, i16* %v22, align 2
  %v24 = sext i16 %v23 to i32
  %v25 = load i16, i16* %v19, align 2
  %v26 = sext i16 %v25 to i32
  %v27 = tail call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %v24, i32 %v26)
  %v28 = shl i32 %v27, 16
  %v29 = ashr exact i32 %v28, 16
  %v30 = getelementptr inbounds i16, i16* %a3, i32 %v21
  %v31 = load i16, i16* %v30, align 2
  %v32 = sext i16 %v31 to i32
  %v33 = tail call i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32 %v29, i32 %v32)
  %v34 = tail call i32 @llvm.hexagon.M2.hmmpyl.s1(i32 %v33, i32 %v29)
  %v35 = tail call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v34, i32 10)
  %v36 = tail call i32 @llvm.hexagon.A2.addsat(i32 %v20, i32 %v35)
  %v37 = add i32 %v21, 1
  %v38 = getelementptr inbounds i16, i16* %v18, i32 2
  %v39 = icmp eq i32 %v37, 3
  br i1 %v39, label %b4, label %b3

b4:                                               ; preds = %b3
  %v40 = tail call i32 @llvm.hexagon.A2.subsat(i32 %v36, i32 %v5)
  %v41 = icmp slt i32 %v40, 0
  br i1 %v41, label %b5, label %b6

b5:                                               ; preds = %b4
  %v42 = tail call i32 @llvm.hexagon.A2.addsat(i32 %v36, i32 0)
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v43 = phi i16 [ %v6, %b5 ], [ %v7, %b4 ]
  %v44 = phi i32 [ %v42, %b5 ], [ %v5, %b4 ]
  %v45 = add i16 %v6, 1
  %v46 = icmp eq i16 %v45, %a5
  br i1 %v46, label %b7, label %b2

b7:                                               ; preds = %b6, %b0
  %v47 = phi i16 [ 0, %b0 ], [ %v43, %b6 ]
  ret i16 %v47
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.sat.ll.s1(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.hmmpyl.s1(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.addsat(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subsat(i32, i32) #1

attributes #0 = { nounwind readonly "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
