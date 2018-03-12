; RUN: llc -march=hexagon -enable-pipeliner -hexagon-expand-condsets=0 < %s
; REQUIRES: asserts

; Disable expand-condsets because it will assert on undefined registers.

; Another test that the pipeliner doesn't ICE when reusing a
; PHI in the epilog code.

@g0 = external global [18 x i16], align 8

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asl.r.r.sat(i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sxth(i32) #0

; Function Attrs: nounwind
define void @f0() #1 {
b0:
  %v0 = alloca [166 x i32], align 8
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = icmp eq i16 undef, 0
  br i1 %v1, label %b2, label %b1

b2:                                               ; preds = %b1
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b3, %b2
  %v2 = add i32 0, 2
  br i1 undef, label %b3, label %b4

b4:                                               ; preds = %b3, %b2
  %v3 = phi i32* [ undef, %b2 ], [ undef, %b3 ]
  %v4 = phi i32 [ 0, %b2 ], [ %v2, %b3 ]
  %v5 = getelementptr [18 x i16], [18 x i16]* @g0, i32 0, i32 undef
  br label %b5

b5:                                               ; preds = %b5, %b4
  %v6 = phi i16 [ 0, %b4 ], [ %v17, %b5 ]
  %v7 = phi i16 [ undef, %b4 ], [ %v6, %b5 ]
  %v8 = phi i32 [ %v4, %b4 ], [ %v35, %b5 ]
  %v9 = phi i32* [ %v3, %b4 ], [ undef, %b5 ]
  %v10 = phi i16* [ undef, %b4 ], [ %v12, %b5 ]
  %v11 = add i32 %v8, 0
  %v12 = getelementptr inbounds i16, i16* %v10, i32 1
  %v13 = sext i16 %v7 to i32
  %v14 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 0, i32 %v13, i32 undef)
  %v15 = getelementptr inbounds i16, i16* %v10, i32 2
  %v16 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v14, i32 undef, i32 undef)
  %v17 = load i16, i16* %v15, align 2, !tbaa !0
  %v18 = sext i16 %v17 to i32
  %v19 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 %v16, i32 %v18, i32 undef)
  %v20 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v19, i32 undef)
  %v21 = getelementptr [166 x i32], [166 x i32]* %v0, i32 0, i32 %v11
  %v22 = load i32, i32* %v21, align 4
  %v23 = call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 %v22, i32 undef)
  %v24 = call i64 @llvm.hexagon.S2.asr.i.p(i64 %v23, i32 15)
  %v25 = call i32 @llvm.hexagon.A2.sat(i64 %v24)
  %v26 = call i32 @llvm.hexagon.A2.subsat(i32 %v20, i32 %v25)
  %v27 = load i16, i16* %v5, align 4
  %v28 = sext i16 %v27 to i32
  %v29 = call i32 @llvm.hexagon.A2.sxth(i32 %v28)
  %v30 = call i64 @llvm.hexagon.M2.dpmpyss.s0(i32 undef, i32 %v29)
  %v31 = call i64 @llvm.hexagon.S2.asr.i.p(i64 %v30, i32 15)
  %v32 = call i32 @llvm.hexagon.A2.sat(i64 %v31)
  %v33 = call i32 @llvm.hexagon.A2.subsat(i32 %v26, i32 %v32)
  %v34 = call i32 @llvm.hexagon.S2.asl.r.r.sat(i32 %v33, i32 undef)
  store i32 %v34, i32* %v9, align 4, !tbaa !4
  %v35 = add i32 %v8, 1
  %v36 = icmp eq i32 %v35, 164
  br i1 %v36, label %b6, label %b5

b6:                                               ; preds = %b5
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32, i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sat(i64) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.asr.i.p(i64, i32) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.dpmpyss.s0(i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subsat(i32, i32) #0

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv55" }
attributes #2 = { noreturn nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"long", !2, i64 0}
