; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: vcmp{{.*}}

target triple = "hexagon"

%s.0 = type { i16, i16, i16, [4 x ptr], i32, i32, i32, ptr, %s.3, i16, i16, i16, i16, i16, %s.4 }
%s.1 = type { ptr, ptr }
%s.2 = type { i16, i16 }
%s.3 = type { i32, ptr, ptr, ptr }
%s.4 = type { i8 }

@g0 = private unnamed_addr constant [7 x i8] c"Static\00", align 1
@g1 = private unnamed_addr constant [5 x i8] c"Heap\00", align 1
@g2 = private unnamed_addr constant [6 x i8] c"Stack\00", align 1

; Function Attrs: nounwind
define i32 @f0(i32 %a0, ptr nocapture %a1) #0 {
b0:
  %v0 = alloca [1 x %s.0], align 8
  %v1 = call i32 @f1(i32 5) #0
  %v2 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 6
  %v3 = icmp eq i32 %v1, 0
  %v4 = select i1 %v3, i32 7, i32 %v1
  store i32 %v4, ptr %v2, align 8, !tbaa !0
  %v7 = load i32, ptr %v0, align 8
  %v8 = trunc i32 %v7 to i16
  %v9 = icmp eq i16 %v8, 0
  br i1 %v9, label %b1, label %b4

b1:                                               ; preds = %b0
  %v10 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 1
  %v11 = icmp ult i32 %v7, 65536
  br i1 %v11, label %b2, label %b4

b2:                                               ; preds = %b1
  %v12 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 2
  %v13 = load i16, ptr %v12, align 4, !tbaa !4
  %v14 = icmp eq i16 %v13, 0
  br i1 %v14, label %b3, label %b4

b3:                                               ; preds = %b2
  store i16 0, ptr %v0, align 8, !tbaa !4
  store i16 0, ptr %v10, align 2, !tbaa !4
  store i16 102, ptr %v12, align 4, !tbaa !4
  br label %b4

b4:                                               ; preds = %b3, %b2, %b1, %b0
  %v15 = phi i16 [ 0, %b3 ], [ 0, %b2 ], [ 0, %b1 ], [ %v8, %b0 ]
  %v16 = insertelement <1 x i32> undef, i32 %v4, i32 0
  %v17 = shufflevector <1 x i32> %v16, <1 x i32> undef, <2 x i32> zeroinitializer
  %v18 = and <2 x i32> %v17, <i32 1, i32 2>
  %v19 = icmp ne <2 x i32> %v18, zeroinitializer
  %v20 = zext <2 x i1> %v19 to <2 x i16>
  %v21 = extractelement <2 x i16> %v20, i32 0
  %v22 = extractelement <2 x i16> %v20, i32 1
  %v23 = add i16 %v21, %v22
  %v24 = lshr i32 %v4, 2
  %v25 = trunc i32 %v24 to i16
  %v26 = and i16 %v25, 1
  %v27 = add i16 %v23, %v26
  %v28 = getelementptr [1 x %s.0], ptr %v0, i32 0, i32 0, i32 4
  %v29 = load i32, ptr %v28, align 8
  %v30 = zext i16 %v27 to i32
  %v31 = udiv i32 %v29, %v30
  store i32 %v31, ptr %v28, align 8
  %v32 = getelementptr [1 x %s.0], ptr %v0, i32 0, i32 0, i32 3, i32 0
  %v33 = and i32 %v4, 1
  %v34 = icmp eq i32 %v33, 0
  br i1 %v34, label %b5, label %b12

b5:                                               ; preds = %b12, %b4
  %v35 = phi i16 [ 0, %b4 ], [ 1, %b12 ]
  %v36 = and i32 %v4, 2
  %v37 = icmp eq i32 %v36, 0
  br i1 %v37, label %b14, label %b13

b6:                                               ; preds = %b16
  %v38 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 3, i32 1
  %v39 = load ptr, ptr %v38, align 4, !tbaa !6
  %v41 = call ptr @f2(i32 %v31, ptr %v39, i16 signext %v15) #0
  %v42 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 7
  store ptr %v41, ptr %v42, align 4, !tbaa !6
  %v43 = load i32, ptr %v2, align 8, !tbaa !0
  br label %b7

b7:                                               ; preds = %b16, %b6
  %v44 = phi i32 [ %v4, %b16 ], [ %v43, %b6 ]
  %v45 = and i32 %v44, 2
  %v46 = icmp eq i32 %v45, 0
  br i1 %v46, label %b9, label %b8

b8:                                               ; preds = %b7
  %v47 = load i32, ptr %v28, align 8, !tbaa !0
  %v48 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 3, i32 2
  %v49 = load ptr, ptr %v48, align 8, !tbaa !6
  %v50 = load i32, ptr %v0, align 8
  %v51 = shl i32 %v50, 16
  %v52 = ashr exact i32 %v51, 16
  %v53 = and i32 %v50, -65536
  %v54 = or i32 %v53, %v52
  %v55 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 8
  %v56 = call i32 @f3(i32 %v47, ptr %v49, i32 %v54, ptr %v55) #0
  %v57 = load i32, ptr %v2, align 8, !tbaa !0
  br label %b9

b9:                                               ; preds = %b8, %b7
  %v58 = phi i32 [ %v44, %b7 ], [ %v57, %b8 ]
  %v59 = and i32 %v58, 4
  %v60 = icmp eq i32 %v59, 0
  br i1 %v60, label %b11, label %b10

b10:                                              ; preds = %b9
  %v61 = load i32, ptr %v28, align 8, !tbaa !0
  %v62 = load i16, ptr %v0, align 8, !tbaa !4
  %v63 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 3, i32 3
  %v64 = load ptr, ptr %v63, align 4, !tbaa !6
  call void @f4(i32 %v61, i16 signext %v62, ptr %v64) #0
  br label %b11

b11:                                              ; preds = %b10, %b9
  ret i32 0

b12:                                              ; preds = %b4
  %v65 = getelementptr [1 x %s.0], ptr %v0, i32 0, i32 0, i32 3, i32 1
  %v66 = load ptr, ptr %v32, align 8
  store ptr %v66, ptr %v65, align 4
  br label %b5

b13:                                              ; preds = %b5
  %v67 = getelementptr [1 x %s.0], ptr %v0, i32 0, i32 0, i32 3, i32 2
  %v68 = load ptr, ptr %v32, align 8
  %v69 = zext i16 %v35 to i32
  %v70 = sub i32 0, %v69
  %v71 = and i32 %v31, %v70
  %v72 = getelementptr inbounds i8, ptr %v68, i32 %v71
  store ptr %v72, ptr %v67, align 8
  %v73 = add i16 %v35, 1
  br label %b14

b14:                                              ; preds = %b13, %b5
  %v74 = phi i16 [ %v35, %b5 ], [ %v73, %b13 ]
  %v75 = and i32 %v4, 4
  %v76 = icmp eq i32 %v75, 0
  br i1 %v76, label %b16, label %b15

b15:                                              ; preds = %b14
  %v77 = getelementptr [1 x %s.0], ptr %v0, i32 0, i32 0, i32 3, i32 3
  %v78 = load ptr, ptr %v32, align 8
  %v79 = zext i16 %v74 to i32
  %v80 = mul i32 %v31, %v79
  %v81 = getelementptr inbounds i8, ptr %v78, i32 %v80
  store ptr %v81, ptr %v77, align 4
  br label %b16

b16:                                              ; preds = %b15, %b14
  br i1 %v34, label %b7, label %b6
}

declare i32 @f1(i32) #0

declare ptr @f2(i32, ptr, i16 signext) #0

declare i32 @f3(i32, ptr, i32, ptr) #0

declare void @f4(i32, i16 signext, ptr) #0

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !2}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !2}
