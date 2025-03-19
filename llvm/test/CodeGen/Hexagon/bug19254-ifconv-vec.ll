; RUN: llc -mtriple=hexagon -O3 < %s | FileCheck %s
; we really just want to be sure this compilation does not abort.
; CHECK: vadd

target triple = "hexagon"

@g0 = private unnamed_addr constant [39 x i8] c"\0AnumTrainingSet =%d  numFeatures = %d\0A\00", align 1

; Function Attrs: nounwind
declare i32 @f0(ptr nocapture readonly, ...) #0

; Function Attrs: nounwind
define void @f1(ptr nocapture readnone %a0, i16 signext %a1, i16 signext %a2, ptr nocapture readnone %a3, ptr nocapture readnone %a4, ptr nocapture %a5, i16 signext %a6, i16 signext %a7) #0 {
b0:
  %v0 = sext i16 %a1 to i32
  %v1 = sext i16 %a2 to i32
  %v2 = tail call i32 (ptr, ...) @f0(ptr @g0, i32 %v0, i32 %v1) #2
  %v3 = tail call <32 x i32> @llvm.hexagon.V6.vd0.128B()
  br label %b1

b1:                                               ; preds = %b18, %b0
  %v4 = phi i32 [ 0, %b0 ], [ %v57, %b18 ]
  %v5 = phi <32 x i32> [ %v3, %b0 ], [ %v56, %b18 ]
  %v6 = icmp slt i32 %v4, %v1
  br i1 %v6, label %b2, label %b3

b2:                                               ; preds = %b1
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> undef, i32 16)
  %v8 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v5, <32 x i32> %v7)
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v9 = phi <32 x i32> [ %v8, %b2 ], [ %v5, %b1 ]
  %v10 = add nuw nsw i32 %v4, 1
  %v11 = icmp slt i32 %v10, %v1
  br i1 %v11, label %b5, label %b6

b4:                                               ; preds = %b18
  %v12 = sext i16 %a6 to i32
  %v13 = tail call double @f3(double 1.000000e+00, i32 %v12) #2
  %v14 = fptosi double %v13 to i32
  %v15 = mul nsw i32 %v0, 2
  %v16 = sitofp i32 %v15 to double
  %v17 = tail call double @f3(double 1.000000e+00, i32 %v12) #2
  %v18 = fmul double %v16, %v17
  %v19 = fptosi double %v18 to i32
  %v20 = tail call i32 @f2(i32 %v14, i32 %v19, i16 signext %a6) #2
  %v21 = extractelement <32 x i32> %v56, i32 0
  %v22 = mul nsw i32 %v20, %v21
  %v23 = trunc i32 %v22 to i16
  store i16 %v23, ptr %a5, align 2, !tbaa !0
  ret void

b5:                                               ; preds = %b3
  %v24 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> undef, i32 16)
  %v25 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v9, <32 x i32> %v24)
  br label %b6

b6:                                               ; preds = %b5, %b3
  %v26 = phi <32 x i32> [ %v25, %b5 ], [ %v9, %b3 ]
  %v27 = add nsw i32 %v4, 2
  %v28 = icmp slt i32 %v27, %v1
  br i1 %v28, label %b7, label %b8

b7:                                               ; preds = %b6
  %v29 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> undef, i32 16)
  %v30 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v26, <32 x i32> %v29)
  br label %b8

b8:                                               ; preds = %b7, %b6
  %v31 = phi <32 x i32> [ %v30, %b7 ], [ %v26, %b6 ]
  %v32 = add nsw i32 %v4, 3
  %v33 = icmp slt i32 %v32, %v1
  br i1 %v33, label %b9, label %b10

b9:                                               ; preds = %b8
  %v34 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> undef, i32 16)
  %v35 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v31, <32 x i32> %v34)
  br label %b10

b10:                                              ; preds = %b9, %b8
  %v36 = phi <32 x i32> [ %v35, %b9 ], [ %v31, %b8 ]
  %v37 = add nsw i32 %v4, 4
  %v38 = icmp slt i32 %v37, %v1
  br i1 %v38, label %b11, label %b12

b11:                                              ; preds = %b10
  %v39 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> undef, i32 16)
  %v40 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v36, <32 x i32> %v39)
  br label %b12

b12:                                              ; preds = %b11, %b10
  %v41 = phi <32 x i32> [ %v40, %b11 ], [ %v36, %b10 ]
  %v42 = add nsw i32 %v4, 5
  %v43 = icmp slt i32 %v42, %v1
  br i1 %v43, label %b13, label %b14

b13:                                              ; preds = %b12
  %v44 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> undef, i32 16)
  %v45 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v41, <32 x i32> %v44)
  br label %b14

b14:                                              ; preds = %b13, %b12
  %v46 = phi <32 x i32> [ %v45, %b13 ], [ %v41, %b12 ]
  %v47 = add nsw i32 %v4, 6
  %v48 = icmp slt i32 %v47, %v1
  br i1 %v48, label %b15, label %b16

b15:                                              ; preds = %b14
  %v49 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> undef, i32 16)
  %v50 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v46, <32 x i32> %v49)
  br label %b16

b16:                                              ; preds = %b15, %b14
  %v51 = phi <32 x i32> [ %v50, %b15 ], [ %v46, %b14 ]
  %v52 = add nsw i32 %v4, 7
  %v53 = icmp slt i32 %v52, %v1
  br i1 %v53, label %b17, label %b18

b17:                                              ; preds = %b16
  %v54 = tail call <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32> undef, i32 16)
  %v55 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v51, <32 x i32> %v54)
  br label %b18

b18:                                              ; preds = %b17, %b16
  %v56 = phi <32 x i32> [ %v55, %b17 ], [ %v51, %b16 ]
  %v57 = add nsw i32 %v4, 8
  %v58 = icmp eq i32 %v57, 64
  br i1 %v58, label %b4, label %b1
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vd0.128B() #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vasrh.128B(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32>, <32 x i32>) #1

declare i32 @f2(i32, i32, i16 signext) #0

declare double @f3(double, i32)

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
