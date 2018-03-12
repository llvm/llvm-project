; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK-NOT: if {{.*}} cmp{{.*}}jump

%s.0 = type opaque
%s.1 = type { i8*, i8*, %s.2*, i32, [0 x i8] }
%s.2 = type opaque

@g0 = private unnamed_addr constant [29 x i8] c"BUG: failure at %s:%d/%s()!\0A\00", align 1
@g1 = private unnamed_addr constant [11 x i8] c"fs/namei.c\00", align 1
@g2 = private unnamed_addr constant [8 x i8] c"putname\00", align 1
@g3 = private unnamed_addr constant [5 x i8] c"BUG!\00", align 1
@g4 = external global %s.0*, align 4

; Function Attrs: nounwind
define void @f0(%s.1* %a0) #0 {
b0:
  %v0 = alloca %s.1*, align 4
  store %s.1* %a0, %s.1** %v0, align 4
  br label %b1, !llvm.loop !0

b1:                                               ; preds = %b0
  %v1 = load %s.1*, %s.1** %v0, align 4
  %v2 = getelementptr inbounds %s.1, %s.1* %v1, i32 0, i32 3
  %v3 = load i32, i32* %v2, align 4
  %v4 = icmp sle i32 %v3, 0
  %v5 = xor i1 %v4, true
  %v6 = xor i1 %v5, true
  %v7 = zext i1 %v6 to i32
  %v8 = call i32 @llvm.expect.i32(i32 %v7, i32 0)
  %v9 = icmp ne i32 %v8, 0
  br i1 %v9, label %b2, label %b5

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b2
  %v10 = call i32 (i8*, ...) @f1(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @g0, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @g1, i32 0, i32 0), i32 246, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @g2, i32 0, i32 0))
  call void (i8*, ...) @f2(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @g3, i32 0, i32 0))
  unreachable

b4:                                               ; No predecessors!
  br label %b5

b5:                                               ; preds = %b4, %b1
  br label %b6

b6:                                               ; preds = %b5
  %v11 = load %s.1*, %s.1** %v0, align 4
  %v12 = getelementptr inbounds %s.1, %s.1* %v11, i32 0, i32 3
  %v13 = load i32, i32* %v12, align 4
  %v14 = add i32 %v13, -1
  store i32 %v14, i32* %v12, align 4
  %v15 = icmp sgt i32 %v14, 0
  br i1 %v15, label %b7, label %b8

b7:                                               ; preds = %b6
  br label %b11

b8:                                               ; preds = %b6
  %v16 = load %s.1*, %s.1** %v0, align 4
  %v17 = getelementptr inbounds %s.1, %s.1* %v16, i32 0, i32 0
  %v18 = load i8*, i8** %v17, align 4
  %v19 = load %s.1*, %s.1** %v0, align 4
  %v20 = getelementptr inbounds %s.1, %s.1* %v19, i32 0, i32 4
  %v21 = getelementptr inbounds [0 x i8], [0 x i8]* %v20, i32 0, i32 0
  %v22 = icmp ne i8* %v18, %v21
  br i1 %v22, label %b9, label %b10

b9:                                               ; preds = %b8
  %v23 = load %s.0*, %s.0** @g4, align 4
  %v24 = load %s.1*, %s.1** %v0, align 4
  %v25 = getelementptr inbounds %s.1, %s.1* %v24, i32 0, i32 0
  %v26 = load i8*, i8** %v25, align 4
  call void @f3(%s.0* %v23, i8* %v26)
  %v27 = load %s.1*, %s.1** %v0, align 4
  %v28 = bitcast %s.1* %v27 to i8*
  call void @f4(i8* %v28)
  br label %b11

b10:                                              ; preds = %b8
  %v29 = load %s.0*, %s.0** @g4, align 4
  %v30 = load %s.1*, %s.1** %v0, align 4
  %v31 = bitcast %s.1* %v30 to i8*
  call void @f3(%s.0* %v29, i8* %v31)
  br label %b11

b11:                                              ; preds = %b10, %b9, %b7
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.expect.i32(i32, i32) #1

; Function Attrs: nounwind
declare i32 @f1(i8*, ...) #0

; Function Attrs: noreturn
declare void @f2(i8*, ...) #2

; Function Attrs: nounwind
declare void @f3(%s.0*, i8*) #0

; Function Attrs: nounwind
declare void @f4(i8*) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { noreturn }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.threadify", i32 101214632}
