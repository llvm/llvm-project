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

declare i32 @f0(ptr, ptr)

; Function Attrs: nounwind
define i32 @f1(i32 %a0, ptr %a1) #0 {
b0:
  %v01 = alloca i32, align 4
  %v12 = alloca i32, align 4
  %v23 = alloca ptr, align 4
  %v34 = alloca i32, align 4
  store i32 0, ptr %v01
  store i32 %a0, ptr %v12, align 4
  store ptr %a1, ptr %v23, align 4
  %v45 = load ptr, ptr %v23, align 4
  %v56 = getelementptr inbounds ptr, ptr %v45, i32 1
  %v67 = load ptr, ptr %v56, align 4
  %v78 = call i32 @f2(ptr %v67)
  store i32 %v78, ptr %v34, align 4
  %v89 = load i32, ptr %v34, align 4
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
  %v11 = call i32 @f0(ptr @g0, ptr @g1)
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v1211 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 1
  %v14 = call i32 @f0(ptr @g0, ptr %v1211)
  br label %b3

b3:                                               ; preds = %b2, %b0
  %v15 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 2
  %v17 = call i32 @f0(ptr @g0, ptr %v15)
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v18 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 3
  %v20 = call i32 @f0(ptr @g0, ptr %v18)
  br label %b5

b5:                                               ; preds = %b4, %b0
  %v21 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 4
  %v2312 = call i32 @f0(ptr @g0, ptr %v21)
  br label %b6

b6:                                               ; preds = %b5, %b0
  %v24 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 5
  %v26 = call i32 @f0(ptr @g0, ptr %v24)
  br label %b7

b7:                                               ; preds = %b6, %b0
  %v27 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 6
  %v29 = call i32 @f0(ptr @g0, ptr %v27)
  br label %b8

b8:                                               ; preds = %b7, %b0
  %v30 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 7
  %v32 = call i32 @f0(ptr @g0, ptr %v30)
  br label %b9

b9:                                               ; preds = %b8, %b0
  %v33 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 8
  %v35 = call i32 @f0(ptr @g0, ptr %v33)
  br label %b10

b10:                                              ; preds = %b9, %b0
  %v36 = getelementptr inbounds [10 x [10 x i8]], ptr @g1, i32 0, i32 9
  %v38 = call i32 @f0(ptr @g0, ptr %v36)
  br label %b11

b11:                                              ; preds = %b10, %b0
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @f2(ptr) #0

attributes #0 = { nounwind }
