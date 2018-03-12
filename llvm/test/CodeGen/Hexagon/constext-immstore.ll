; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

%s.0 = type { i8, i8, i8*, i8, i32, %s.0*, %s.0* }
%s.1 = type { %s.1*, %s.2, %s.0*, %s.2 }
%s.2 = type { i8, %s.3, i8 }
%s.3 = type { %s.4* }
%s.4 = type { [65 x i8], i16, %s.4*, %s.4* }

@g0 = private unnamed_addr constant [4 x i8] c"and\00", align 1
@g1 = private unnamed_addr constant [3 x i8] c"or\00", align 1
@g2 = private unnamed_addr constant [8 x i8] c"implies\00", align 1
@g3 = private unnamed_addr constant [3 x i8] c"if\00", align 1
@g4 = global [4 x %s.0] [%s.0 { i8 1, i8 38, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8 1, i32 8, %s.0* null, %s.0* null }, %s.0 { i8 2, i8 124, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @g1, i32 0, i32 0), i8 1, i32 7, %s.0* null, %s.0* null }, %s.0 { i8 3, i8 62, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @g2, i32 0, i32 0), i8 1, i32 1, %s.0* null, %s.0* null }, %s.0 { i8 4, i8 60, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @g3, i32 0, i32 0), i8 1, i32 1, %s.0* null, %s.0* null }], align 8
@g5 = internal global [64 x i8] zeroinitializer, align 8
@g6 = internal unnamed_addr global %s.0* null, align 4

; Function Attrs: nounwind
define %s.1* @f0() #0 {
b0:
  %v0 = tail call %s.1* @f1(%s.1* null) #0
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = tail call zeroext i8 @f2(i8* getelementptr inbounds ([64 x i8], [64 x i8]* @g5, i32 0, i32 0)) #0
  switch i8 %v1, label %b1 [
    i8 8, label %b2
    i8 6, label %b2
  ]

b2:                                               ; preds = %b1, %b1
  ret %s.1* %v0
}

declare %s.1* @f1(%s.1*) #0

declare zeroext i8 @f2(i8*) #0

; Function Attrs: nounwind
define void @f3() #0 {
b0:
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 0), %s.0** @g6, align 4
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 0), %s.0** getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 0, i32 5), align 8
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 0), %s.0** getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 0, i32 6), align 4
  %v0 = load i8*, i8** getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 1, i32 2), align 4
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi %s.0* [ getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 0), %b0 ], [ %v9, %b1 ]
  %v2 = getelementptr inbounds %s.0, %s.0* %v1, i32 0, i32 2
  %v3 = load i8*, i8** %v2, align 4
  %v4 = tail call i32 @f4(i8* %v0, i8* %v3) #0
  %v5 = icmp sgt i32 %v4, 0
  %v6 = getelementptr inbounds %s.0, %s.0* %v1, i32 0, i32 5
  %v7 = getelementptr inbounds %s.0, %s.0* %v1, i32 0, i32 6
  %v8 = select i1 %v5, %s.0** %v6, %s.0** %v7
  %v9 = load %s.0*, %s.0** %v8, align 4
  %v10 = icmp eq %s.0* %v9, null
  br i1 %v10, label %b2, label %b1

b2:                                               ; preds = %b1
  %v11 = phi i32 [ %v4, %b1 ]
  %v12 = phi %s.0* [ %v1, %b1 ]
  %v13 = icmp sgt i32 %v11, 0
  br i1 %v13, label %b3, label %b4

b3:                                               ; preds = %b2
  %v14 = getelementptr inbounds %s.0, %s.0* %v12, i32 0, i32 5
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 1), %s.0** %v14, align 4
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v15 = getelementptr inbounds %s.0, %s.0* %v12, i32 0, i32 6
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 1), %s.0** %v15, align 4
  %v16 = load %s.0*, %s.0** @g6, align 4
  %v17 = icmp eq %s.0* %v16, null
  br i1 %v17, label %b8, label %b5

b5:                                               ; preds = %b4
  %v18 = load i8*, i8** getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 2, i32 2), align 4
  br label %b6

b6:                                               ; preds = %b6, %b5
  %v19 = phi %s.0* [ %v16, %b5 ], [ %v27, %b6 ]
  %v20 = getelementptr inbounds %s.0, %s.0* %v19, i32 0, i32 2
  %v21 = load i8*, i8** %v20, align 4
  %v22 = tail call i32 @f4(i8* %v18, i8* %v21) #0
  %v23 = icmp sgt i32 %v22, 0
  %v24 = getelementptr inbounds %s.0, %s.0* %v19, i32 0, i32 5
  %v25 = getelementptr inbounds %s.0, %s.0* %v19, i32 0, i32 6
  %v26 = select i1 %v23, %s.0** %v24, %s.0** %v25
  %v27 = load %s.0*, %s.0** %v26, align 4
  %v28 = icmp eq %s.0* %v27, null
  br i1 %v28, label %b7, label %b6

b7:                                               ; preds = %b6
  %v29 = phi i32 [ %v22, %b6 ]
  %v30 = phi %s.0* [ %v19, %b6 ]
  br label %b8

b8:                                               ; preds = %b7, %b4
  %v31 = phi i32 [ %v11, %b4 ], [ %v29, %b7 ]
  %v32 = phi %s.0* [ null, %b4 ], [ %v30, %b7 ]
  %v33 = icmp sgt i32 %v31, 0
  br i1 %v33, label %b9, label %b10

b9:                                               ; preds = %b8
  %v34 = getelementptr inbounds %s.0, %s.0* %v32, i32 0, i32 5
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 2), %s.0** %v34, align 4
  br label %b10

b10:                                              ; preds = %b9, %b8
  %v35 = getelementptr inbounds %s.0, %s.0* %v32, i32 0, i32 6
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 2), %s.0** %v35, align 4
  %v36 = load %s.0*, %s.0** @g6, align 4
  %v37 = icmp eq %s.0* %v36, null
  br i1 %v37, label %b14, label %b11

b11:                                              ; preds = %b10
  %v38 = load i8*, i8** getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 3, i32 2), align 4
  br label %b12

b12:                                              ; preds = %b12, %b11
  %v39 = phi %s.0* [ %v36, %b11 ], [ %v47, %b12 ]
  %v40 = getelementptr inbounds %s.0, %s.0* %v39, i32 0, i32 2
  %v41 = load i8*, i8** %v40, align 4
  %v42 = tail call i32 @f4(i8* %v38, i8* %v41) #0
  %v43 = icmp sgt i32 %v42, 0
  %v44 = getelementptr inbounds %s.0, %s.0* %v39, i32 0, i32 5
  %v45 = getelementptr inbounds %s.0, %s.0* %v39, i32 0, i32 6
  %v46 = select i1 %v43, %s.0** %v44, %s.0** %v45
  %v47 = load %s.0*, %s.0** %v46, align 4
  %v48 = icmp eq %s.0* %v47, null
  br i1 %v48, label %b13, label %b12

b13:                                              ; preds = %b12
  %v49 = phi i32 [ %v42, %b12 ]
  %v50 = phi %s.0* [ %v39, %b12 ]
  br label %b14

b14:                                              ; preds = %b13, %b10
  %v51 = phi i32 [ %v31, %b10 ], [ %v49, %b13 ]
  %v52 = phi %s.0* [ null, %b10 ], [ %v50, %b13 ]
  %v53 = icmp sgt i32 %v51, 0
  br i1 %v53, label %b15, label %b16

b15:                                              ; preds = %b14
  %v54 = getelementptr inbounds %s.0, %s.0* %v52, i32 0, i32 5
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 3), %s.0** %v54, align 4
  br label %b16

b16:                                              ; preds = %b15, %b14
  %v55 = getelementptr inbounds %s.0, %s.0* %v52, i32 0, i32 6
  store %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g4, i32 0, i32 3), %s.0** %v55, align 4
  ret void
}

; Function Attrs: nounwind readonly
declare i32 @f4(i8* nocapture, i8* nocapture) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
