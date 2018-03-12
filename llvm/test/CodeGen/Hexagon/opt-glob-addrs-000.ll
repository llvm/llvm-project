; RUN: llc -march=hexagon -O2 -disable-hexagon-misched < %s | FileCheck %s

target triple = "hexagon-unknown--elf"

; CHECK-LABEL: f1:
; CHECK-DAG:      r16 = ##.Lg0+32767
; CHECK-DAG:      r17 = ##g1+32767

; CHECK-LABEL: LBB0_2:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32767)
; CHECK:        }

; CHECK-LABEL: LBB0_3:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32757)
; CHECK:        }

; CHECK-LABEL: LBB0_4:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32747)
; CHECK:        }

; CHECK-LABEL: LBB0_5:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32737)
; CHECK:        }

; CHECK-LABEL: LBB0_6:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32727)
; CHECK:        }

; CHECK-LABEL: LBB0_7:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32717)
; CHECK:        }

; CHECK-LABEL: LBB0_8:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32707)
; CHECK:        }

; CHECK-LABEL: LBB0_9:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32697)
; CHECK:        }

; CHECK-LABEL: LBB0_10:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32687)
; CHECK:        }

; CHECK-LABEL: LBB0_11:
; CHECK:        {
; CHECK-DAG:      call f0
; CHECK-DAG:      r0 = add(r16,#-32767)
; CHECK-DAG:      r1 = add(r17,#-32677)
; CHECK:        }

@g0 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@g1 = internal constant [10 x [10 x i8]] [[10 x i8] c"[0000]\00\00\00\00", [10 x i8] c"[0001]\00\00\00\00", [10 x i8] c"[0002]\00\00\00\00", [10 x i8] c"[0003]\00\00\00\00", [10 x i8] c"[0004]\00\00\00\00", [10 x i8] c"[0005]\00\00\00\00", [10 x i8] c"[0006]\00\00\00\00", [10 x i8] c"[0007]\00\00\00\00", [10 x i8] c"[0008]\00\00\00\00", [10 x i8] c"[0009]\00\00\00\00"], align 16

declare i32 @f0(i8*, i8*)

; Function Attrs: nounwind
define i32 @f1(i32 %a0, i8** %a1) #0 {
b0:
  %v01 = alloca i32, align 4
  %v12 = alloca i32, align 4
  %v23 = alloca i8**, align 4
  %v34 = alloca i32, align 4
  store i32 0, i32* %v01
  store i32 %a0, i32* %v12, align 4
  store i8** %a1, i8*** %v23, align 4
  %v45 = load i8**, i8*** %v23, align 4
  %v56 = getelementptr inbounds i8*, i8** %v45, i32 1
  %v67 = load i8*, i8** %v56, align 4
  %v78 = call i32 @f2(i8* %v67)
  store i32 %v78, i32* %v34, align 4
  %v89 = load i32, i32* %v34, align 4
  switch i32 %v89, label %b11 [
    i32 0, label %b1
    i32 1, label %b2
    i32 2, label %b3
    i32 3, label %b4
    i32 4, label %b5
    i32 5, label %b6
    i32 6, label %b7
    i32 7, label %b8
    i32 8, label %b9
    i32 9, label %b10
  ]

b1:                                               ; preds = %b0
  %v910 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 0
  %v10 = getelementptr inbounds [10 x i8], [10 x i8]* %v910, i32 0, i32 0
  %v11 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v10)
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v1211 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 1
  %v13 = getelementptr inbounds [10 x i8], [10 x i8]* %v1211, i32 0, i32 0
  %v14 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v13)
  br label %b3

b3:                                               ; preds = %b2, %b0
  %v15 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 2
  %v16 = getelementptr inbounds [10 x i8], [10 x i8]* %v15, i32 0, i32 0
  %v17 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v16)
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v18 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 3
  %v19 = getelementptr inbounds [10 x i8], [10 x i8]* %v18, i32 0, i32 0
  %v20 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v19)
  br label %b5

b5:                                               ; preds = %b4, %b0
  %v21 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 4
  %v22 = getelementptr inbounds [10 x i8], [10 x i8]* %v21, i32 0, i32 0
  %v2312 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v22)
  br label %b6

b6:                                               ; preds = %b5, %b0
  %v24 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 5
  %v25 = getelementptr inbounds [10 x i8], [10 x i8]* %v24, i32 0, i32 0
  %v26 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v25)
  br label %b7

b7:                                               ; preds = %b6, %b0
  %v27 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 6
  %v28 = getelementptr inbounds [10 x i8], [10 x i8]* %v27, i32 0, i32 0
  %v29 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v28)
  br label %b8

b8:                                               ; preds = %b7, %b0
  %v30 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 7
  %v31 = getelementptr inbounds [10 x i8], [10 x i8]* %v30, i32 0, i32 0
  %v32 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v31)
  br label %b9

b9:                                               ; preds = %b8, %b0
  %v33 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 8
  %v3413 = getelementptr inbounds [10 x i8], [10 x i8]* %v33, i32 0, i32 0
  %v35 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v3413)
  br label %b10

b10:                                              ; preds = %b9, %b0
  %v36 = getelementptr inbounds [10 x [10 x i8]], [10 x [10 x i8]]* @g1, i32 0, i32 9
  %v37 = getelementptr inbounds [10 x i8], [10 x i8]* %v36, i32 0, i32 0
  %v38 = call i32 @f0(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* %v37)
  br label %b11

b11:                                              ; preds = %b10, %b0
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @f2(i8*) #0

attributes #0 = { nounwind }
