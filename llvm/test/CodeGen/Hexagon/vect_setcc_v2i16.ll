; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts
; CHECK: f0

target triple = "hexagon"

@g0 = internal unnamed_addr global [24 x i16] zeroinitializer, align 8

; Function Attrs: nounwind
define void @f0(i16* nocapture %a0) #0 {
b0:
  %v0 = alloca [128 x i16], align 8
  %v1 = alloca [16 x i16], align 8
  %v2 = bitcast [128 x i16]* %v0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 256, i8* %v2) #2
  %v3 = getelementptr [128 x i16], [128 x i16]* %v0, i32 0, i32 80
  br label %b8

b1:                                               ; preds = %b3
  br label %b2

b2:                                               ; preds = %b4, %b1
  call void @llvm.lifetime.end.p0i8(i64 256, i8* %v2) #2
  ret void

b3:                                               ; preds = %b5, %b3
  %v4 = phi i16* [ %v26, %b5 ], [ %v9, %b3 ]
  %v5 = phi i32 [ 0, %b5 ], [ %v7, %b3 ]
  %v6 = bitcast i16* %v4 to <4 x i16>*
  store <4 x i16> <i16 1, i16 1, i16 1, i16 1>, <4 x i16>* %v6, align 8
  %v7 = add nsw i32 %v5, 4
  %v8 = icmp slt i32 %v5, 12
  %v9 = getelementptr i16, i16* %v4, i32 4
  br i1 %v8, label %b3, label %b1

b4:                                               ; preds = %b6
  %v10 = getelementptr [16 x i16], [16 x i16]* %v1, i32 0, i32 13
  %v11 = bitcast i16* %v10 to <2 x i16>*
  %v12 = load <2 x i16>, <2 x i16>* %v11, align 2
  %v13 = icmp sgt <2 x i16> %v12, <i16 11, i16 11>
  %v14 = zext <2 x i1> %v13 to <2 x i32>
  %v15 = add <2 x i32> %v39, %v14
  %v16 = add <2 x i32> %v15, %v40
  %v17 = extractelement <2 x i32> %v16, i32 0
  %v18 = extractelement <2 x i32> %v16, i32 1
  %v19 = getelementptr [16 x i16], [16 x i16]* %v1, i32 0, i32 15
  %v20 = load i16, i16* %v19, align 2
  %v21 = icmp sgt i16 %v20, 11
  %v22 = zext i1 %v21 to i32
  %v23 = add i32 %v18, %v22
  %v24 = add i32 %v23, %v17
  %v25 = icmp slt i32 %v24, 5
  br i1 %v25, label %b5, label %b2

b5:                                               ; preds = %b4
  %v26 = getelementptr [16 x i16], [16 x i16]* %v1, i32 0, i32 0
  br label %b3

b6:                                               ; preds = %b7, %b6
  %v27 = phi <2 x i32> [ zeroinitializer, %b7 ], [ %v40, %b6 ]
  %v28 = phi <2 x i32> [ zeroinitializer, %b7 ], [ %v39, %b6 ]
  %v29 = phi i16* [ %v44, %b7 ], [ %v43, %b6 ]
  %v30 = phi i32 [ 0, %b7 ], [ %v41, %b6 ]
  %v31 = bitcast i16* %v29 to <4 x i16>*
  %v32 = load <4 x i16>, <4 x i16>* %v31, align 2
  %v33 = icmp sgt <4 x i16> %v32, <i16 11, i16 11, i16 11, i16 11>
  %v34 = zext <4 x i1> %v33 to <4 x i16>
  %v35 = shufflevector <4 x i16> %v34, <4 x i16> undef, <2 x i32> <i32 2, i32 3>
  %v36 = shufflevector <4 x i16> %v34, <4 x i16> undef, <2 x i32> <i32 0, i32 1>
  %v37 = zext <2 x i16> %v36 to <2 x i32>
  %v38 = zext <2 x i16> %v35 to <2 x i32>
  %v39 = add <2 x i32> %v28, %v37
  %v40 = add <2 x i32> %v27, %v38
  %v41 = add nsw i32 %v30, 4
  %v42 = icmp slt i32 %v30, 4
  %v43 = getelementptr i16, i16* %v29, i32 4
  br i1 %v42, label %b6, label %b4

b7:                                               ; preds = %b8
  %v44 = getelementptr [16 x i16], [16 x i16]* %v1, i32 0, i32 5
  br label %b6

b8:                                               ; preds = %b8, %b0
  %v45 = phi i16* [ %v3, %b0 ], [ %v53, %b8 ]
  %v46 = phi i16* [ getelementptr inbounds ([24 x i16], [24 x i16]* @g0, i32 0, i32 0), %b0 ], [ %v54, %b8 ]
  %v47 = phi i32 [ 0, %b0 ], [ %v51, %b8 ]
  %v48 = bitcast i16* %v45 to <4 x i16>*
  %v49 = load <4 x i16>, <4 x i16>* %v48, align 8
  %v50 = bitcast i16* %v46 to <4 x i16>*
  store <4 x i16> %v49, <4 x i16>* %v50, align 8
  %v51 = add nsw i32 %v47, 4
  %v52 = icmp slt i32 %v47, 20
  %v53 = getelementptr i16, i16* %v45, i32 4
  %v54 = getelementptr i16, i16* %v46, i32 4
  br i1 %v52, label %b8, label %b7
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }
