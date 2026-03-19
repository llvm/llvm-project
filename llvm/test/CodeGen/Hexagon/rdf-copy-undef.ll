; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; After a copy R20 = R29, RDF copy propagation attempted to replace R20 with
; R29. R29 did not have a reaching def at that point (which isn't unusual),
; but copy propagation tried to link the new use of R29 to the presumed
; reaching def (which was null), causing a crash.

target triple = "hexagon"

@g0 = external unnamed_addr global i1, align 4

; Function Attrs: nounwind
declare ptr @llvm.stacksave() #0

; Function Attrs: nounwind
declare void @llvm.stackrestore(ptr) #0

; Function Attrs: norecurse nounwind
declare fastcc void @f0(i16 signext, i16 signext, i16 signext, ptr nocapture readonly, i16 signext, ptr nocapture) unnamed_addr #1

; Function Attrs: norecurse nounwind
declare fastcc signext i16 @f1(i16 signext, i16 signext) unnamed_addr #1

; Function Attrs: norecurse nounwind
define fastcc i32 @f2(ptr nocapture readonly %a0, i16 signext %a1, i16 signext %a2, ptr nocapture readonly %a3, i16 signext %a4, ptr nocapture readonly %a51, ptr nocapture %a6) unnamed_addr #1 {
b0:
  %v0 = tail call ptr @llvm.stacksave()
  %v1 = tail call fastcc signext i16 @f1(i16 signext %a2, i16 signext %a1)
  br i1 undef, label %b7, label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b3, label %b2

b2:                                               ; preds = %b1
  br i1 undef, label %b4, label %b8

b3:                                               ; preds = %b1
  br i1 undef, label %b5, label %b8

b4:                                               ; preds = %b4, %b2
  br i1 undef, label %b4, label %b6, !llvm.loop !2

b5:                                               ; preds = %b5, %b3
  %v2 = phi i16 [ %v3, %b5 ], [ 0, %b3 ]
  %v3 = add i16 %v2, 1
  %v4 = icmp sgt i32 0, -1073741825
  br i1 %v4, label %b5, label %b6

b6:                                               ; preds = %b5, %b4
  %v5 = phi i16 [ %v3, %b5 ], [ undef, %b4 ]
  br label %b7

b7:                                               ; preds = %b6, %b0
  %v6 = phi i16 [ %v5, %b6 ], [ 0, %b0 ]
  br i1 undef, label %b9, label %b8

b8:                                               ; preds = %b7, %b3, %b2
  %v7 = or i32 0, undef
  br label %b9

b9:                                               ; preds = %b8, %b7
  %v8 = phi i16 [ 0, %b8 ], [ %v6, %b7 ]
  %v9 = phi i32 [ %v7, %b8 ], [ 0, %b7 ]
  %v10 = load i16, ptr undef, align 2, !tbaa !4
  %v11 = sext i16 %v10 to i32
  %v12 = zext i16 %v10 to i32
  br i1 undef, label %b10, label %b11

b10:                                              ; preds = %b9
  store i1 true, ptr @g0, align 4
  br label %b11

b11:                                              ; preds = %b10, %b9
  %v13 = load i16, ptr undef, align 2, !tbaa !4
  %v14 = sext i16 %v13 to i32
  %v15 = shl nuw i32 %v12, 16
  %v16 = and i32 %v9, 65535
  %v17 = mul nsw i32 %v11, %v16
  %v18 = sitofp i32 %v15 to double
  %v19 = fsub double %v18, undef
  %v20 = sub nsw i32 %v15, %v17
  %v21 = fptosi double %v19 to i32
  %v22 = select i1 undef, i32 %v21, i32 %v20
  %v23 = mul nsw i32 %v14, %v16
  %v24 = add nsw i32 %v23, %v22
  %v25 = add nsw i32 %v24, 32768
  %v26 = lshr i32 %v25, 16
  %v27 = xor i1 undef, true
  %v28 = and i1 %v27, undef
  br i1 %v28, label %b12, label %b13

b12:                                              ; preds = %b11
  store i1 true, ptr @g0, align 4
  br label %b13

b13:                                              ; preds = %b12, %b11
  br i1 undef, label %b14, label %b24

b14:                                              ; preds = %b13
  br label %b15

b15:                                              ; preds = %b23, %b14
  br i1 undef, label %b16, label %b17

b16:                                              ; preds = %b15
  br label %b19

b17:                                              ; preds = %b15
  %v29 = trunc i32 %v26 to i16
  %v30 = icmp eq i16 %v29, -32768
  %v31 = and i1 undef, %v30
  br i1 %v31, label %b18, label %b19

b18:                                              ; preds = %b17
  store i1 true, ptr @g0, align 4
  br label %b20

b19:                                              ; preds = %b17, %b16
  br label %b20

b20:                                              ; preds = %b19, %b18
  %v32 = phi i32 [ 2147483647, %b18 ], [ 0, %b19 ]
  %v33 = icmp eq i16 %v8, 32767
  br i1 %v33, label %b21, label %b22

b21:                                              ; preds = %b20
  store i1 true, ptr @g0, align 4
  br label %b23

b22:                                              ; preds = %b20
  br label %b23

b23:                                              ; preds = %b22, %b21
  %v34 = add nsw i32 %v32, 32768
  %v35 = lshr i32 %v34, 16
  %v36 = trunc i32 %v35 to i16
  store i16 %v36, ptr undef, align 2, !tbaa !4
  br i1 undef, label %b24, label %b15

b24:                                              ; preds = %b23, %b13
  call fastcc void @f0(i16 signext undef, i16 signext %a1, i16 signext %a2, ptr %a3, i16 signext %a4, ptr %a6)
  call void @llvm.stackrestore(ptr %v0)
  ret i32 undef
}

attributes #0 = { nounwind }
attributes #1 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

!llvm.module.flags = !{!0}

!0 = !{i32 6, !"Target Features", !1}
!1 = !{!"+hvx,+hvx-length64b"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.threadify", i32 43789156}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
