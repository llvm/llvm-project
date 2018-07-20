; RUN: llc -march=hexagon -debug-only=pipeliner < %s -o - 2>&1 | FileCheck %s
; REQUIRES: asserts

; Test that there is a chain edge between two dependent Phis.
; The pipeliner tries to remove chains between unrelated Phis, but
; was too aggressive in some cases. When this happens the two Phis may get
; scheduled too far apart. In this case, the second Phi was scheduled in
; the next stage.

; CHECK: SU([[SU1:[0-9]+]]): %14:intregs = PHI %{{[0-9]+}}:intregs, %bb.0, %{{[0-9]+}}:intregs, %bb.1
; CHECK: Successors:
; CHECK: SU({{.*}}): Data Latency=0
; CHECK: SU([[SU2:[0-9]+]]): Data Latency=0
; CHECK: SU([[SU2]]):   %{{[0-9]+}}:intregs = PHI %{{[0-9]+}}:intregs, %bb.0, %14:intregs, %bb.1
; CHECK: Predecessors:
; CHECK: SU([[SU1]]): Data Latency=0

%s.0 = type { i16, i8, i32, i8*, i8*, i8*, i8*, i8*, i8*, i32*, [2 x i32], i8*, i8*, i8*, %s.1, i8*, [8 x i8], i8 }
%s.1 = type { i32, i16, i16 }
%s.2 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }

@g0 = global i32 0, align 4
@g1 = global i32 0, align 4
@g2 = global i32 0, align 4
@g3 = global i32 0, align 4
@g4 = global i32 0, align 4
@g5 = common global i32 0, align 4
@g6 = external global %s.0
@g7 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: nounwind
declare i32 @f0(%s.0* nocapture, i8* nocapture readonly, ...) #0

; Function Attrs: nounwind
define void @f1(%s.2* nocapture %a0, i32* nocapture readonly %a1, i32* nocapture readonly %a2, i16 signext %a3) #0 {
b0:
  %v0 = load i32, i32* %a2, align 4
  %v1 = tail call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v0, i32 2)
  %v2 = tail call i32 @llvm.hexagon.A2.sath(i32 %v1)
  store i32 0, i32* @g5, align 4
  %v3 = load i32, i32* @g0, align 4
  %v4 = load i32, i32* @g1, align 4
  %v5 = load i32, i32* @g2, align 4
  %v6 = load i32, i32* @g3, align 4
  %v7 = load i32, i32* @g4, align 4
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v8 = phi i32 [ %v7, %b0 ], [ %v52, %b1 ]
  %v9 = phi i32 [ %v6, %b0 ], [ %v50, %b1 ]
  %v10 = phi i32 [ %v5, %b0 ], [ %v46, %b1 ]
  %v11 = phi i32 [ %v4, %b0 ], [ %v44, %b1 ]
  %v12 = phi i32 [ %v3, %b0 ], [ %v38, %b1 ]
  %v13 = phi i32 [ 0, %b0 ], [ %v53, %b1 ]
  %v14 = phi i32* [ %a2, %b0 ], [ %v26, %b1 ]
  %v15 = phi i32* [ %a1, %b0 ], [ %v19, %b1 ]
  %v16 = phi i32 [ %v2, %b0 ], [ %v32, %b1 ]
  %v17 = phi i32 [ 0, %b0 ], [ %v25, %b1 ]
  %v18 = phi i32 [ 0, %b0 ], [ %v16, %b1 ]
  %v19 = getelementptr inbounds i32, i32* %v15, i32 1
  %v20 = load i32, i32* %v15, align 4
  %v21 = tail call i32 @llvm.hexagon.A2.asrh(i32 %v20)
  %v22 = shl i32 %v21, 16
  %v23 = ashr exact i32 %v22, 16
  %v24 = tail call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v23, i32 2)
  %v25 = tail call i32 @llvm.hexagon.A2.sath(i32 %v24)
  %v26 = getelementptr inbounds i32, i32* %v14, i32 1
  %v27 = load i32, i32* %v14, align 4
  %v28 = tail call i32 @llvm.hexagon.A2.asrh(i32 %v27)
  %v29 = shl i32 %v28, 16
  %v30 = ashr exact i32 %v29, 16
  %v31 = tail call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v30, i32 2)
  %v32 = tail call i32 @llvm.hexagon.A2.sath(i32 %v31)
  %v33 = shl i32 %v17, 16
  %v34 = ashr exact i32 %v33, 16
  %v35 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32 %v12, i32 %v34, i32 %v34)
  %v36 = shl i32 %v16, 16
  %v37 = ashr exact i32 %v36, 16
  %v38 = tail call i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32 %v35, i32 %v37, i32 %v37)
  store i32 %v38, i32* @g0, align 4
  %v39 = shl i32 %v25, 16
  %v40 = ashr exact i32 %v39, 16
  %v41 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v11, i32 %v40, i32 %v34)
  %v42 = shl i32 %v32, 16
  %v43 = ashr exact i32 %v42, 16
  %v44 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v41, i32 %v43, i32 %v37)
  store i32 %v44, i32* @g1, align 4
  %v45 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v10, i32 %v43, i32 %v34)
  %v46 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v45, i32 %v40, i32 %v37)
  store i32 %v46, i32* @g2, align 4
  %v47 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v9, i32 %v40, i32 0)
  %v48 = shl i32 %v18, 16
  %v49 = ashr exact i32 %v48, 16
  %v50 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v47, i32 %v43, i32 %v49)
  store i32 %v50, i32* @g3, align 4
  %v51 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v8, i32 %v43, i32 0)
  %v52 = tail call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v51, i32 %v40, i32 %v49)
  store i32 %v52, i32* @g4, align 4
  %v53 = add nsw i32 %v13, 1
  %v54 = icmp slt i32 %v53, 4
  store i32 %v53, i32* @g5, align 4
  br i1 %v54, label %b1, label %b2

b2:                                               ; preds = %b1
  %v55 = tail call i32 (%s.0*, i8*, ...) @f0(%s.0* @g6, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g7, i32 0, i32 0), i32 %v46) #2
  %v56 = load i32, i32* @g2, align 4
  %v57 = getelementptr inbounds %s.2, %s.2* %a0, i32 0, i32 5
  store i32 %v56, i32* %v57, align 4, !tbaa !0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sath(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.asrh(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.nac.sat.ll.s1(i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!0 = !{!1, !2, i64 20}
!1 = !{!"", !2, i64 0, !2, i64 4, !2, i64 8, !2, i64 12, !2, i64 16, !2, i64 20, !2, i64 24, !2, i64 28, !2, i64 32}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
