; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts
; CHECK: f0:

target triple = "hexagon"

; Function Attrs: nounwind readonly
define void @f0(i16* nocapture %a0) #0 {
b0:
  %v0 = alloca [16 x i16], align 8
  %v1 = load i16, i16* %a0, align 2, !tbaa !0
  %v2 = getelementptr [16 x i16], [16 x i16]* %v0, i32 0, i32 5
  br label %b12

b1:                                               ; preds = %b11
  %v3 = icmp slt i16 %v1, 46
  br i1 %v3, label %b3, label %b2

b2:                                               ; preds = %b1
  br label %b5

b3:                                               ; preds = %b1
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v4 = phi i32 [ %v6, %b4 ], [ 0, %b3 ]
  %v5 = getelementptr inbounds [16 x i16], [16 x i16]* %v0, i32 0, i32 %v4
  store i16 1, i16* %v5, align 2, !tbaa !0
  %v6 = add nsw i32 %v4, 1
  %v7 = icmp eq i32 %v6, 16
  br i1 %v7, label %b8, label %b4

b5:                                               ; preds = %b7, %b2
  %v8 = phi i32 [ %v12, %b7 ], [ 0, %b2 ]
  %v9 = getelementptr inbounds [16 x i16], [16 x i16]* %v0, i32 0, i32 %v8
  %v10 = load i16, i16* %v9, align 2, !tbaa !0
  %v11 = icmp slt i16 %v10, 13
  br i1 %v11, label %b6, label %b7

b6:                                               ; preds = %b5
  store i16 1, i16* %v9, align 2, !tbaa !0
  br label %b7

b7:                                               ; preds = %b6, %b5
  %v12 = add nsw i32 %v8, 1
  %v13 = icmp eq i32 %v12, 16
  br i1 %v13, label %b9, label %b5

b8:                                               ; preds = %b4
  br label %b10

b9:                                               ; preds = %b7
  br label %b10

b10:                                              ; preds = %b11, %b9, %b8
  ret void

b11:                                              ; preds = %b12
  %v14 = add <2 x i32> %v31, %v32
  %v15 = extractelement <2 x i32> %v14, i32 0
  %v16 = extractelement <2 x i32> %v14, i32 1
  %v17 = add i32 %v16, %v15
  %v18 = icmp eq i32 %v17, 1
  br i1 %v18, label %b1, label %b10

b12:                                              ; preds = %b12, %b0
  %v19 = phi <2 x i32> [ zeroinitializer, %b0 ], [ %v31, %b12 ]
  %v20 = phi <2 x i32> [ zeroinitializer, %b0 ], [ %v32, %b12 ]
  %v21 = phi i16* [ %v2, %b0 ], [ %v35, %b12 ]
  %v22 = phi i32 [ 0, %b0 ], [ %v33, %b12 ]
  %v23 = bitcast i16* %v21 to <4 x i16>*
  %v24 = load <4 x i16>, <4 x i16>* %v23, align 2
  %v25 = icmp sgt <4 x i16> %v24, <i16 11, i16 11, i16 11, i16 11>
  %v26 = zext <4 x i1> %v25 to <4 x i16>
  %v27 = shufflevector <4 x i16> %v26, <4 x i16> undef, <2 x i32> <i32 2, i32 3>
  %v28 = shufflevector <4 x i16> %v26, <4 x i16> undef, <2 x i32> <i32 0, i32 1>
  %v29 = zext <2 x i16> %v28 to <2 x i32>
  %v30 = zext <2 x i16> %v27 to <2 x i32>
  %v31 = add <2 x i32> %v19, %v29
  %v32 = add <2 x i32> %v20, %v30
  %v33 = add nsw i32 %v22, 4
  %v34 = icmp slt i32 %v22, 4
  %v35 = getelementptr i16, i16* %v21, i32 4
  br i1 %v34, label %b12, label %b11
}

attributes #0 = { nounwind readonly "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
