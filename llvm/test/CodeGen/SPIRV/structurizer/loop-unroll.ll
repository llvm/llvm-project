; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#For:]] "for_loop"
; CHECK-DAG: OpName %[[#While:]] "while_loop"
; CHECK-DAG: OpName %[[#DoWhile:]] "do_while_loop"
; CHECK-DAG: OpName %[[#Disable:]] "unroll_disable"
; CHECK-DAG: OpName %[[#Count:]] "unroll_count"
; CHECK-DAG: OpName %[[#Full:]] "unroll_full"
; CHECK-DAG: OpName %[[#FullCount:]] "unroll_full_count"
; CHECK-DAG: OpName %[[#EnableDisable:]] "unroll_enable_disable"

; CHECK: %[[#For]] = OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll

; CHECK: %[[#While]] = OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll

; CHECK: %[[#DoWhile]] = OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll

; CHECK: %[[#Disable]] = OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] DontUnroll

; CHECK: %[[#Count]] = OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] PartialCount 4

; CHECK: %[[#Full]] = OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll

; CHECK: %[[#FullCount]] = OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] Unroll|PartialCount 4

; CHECK: %[[#EnableDisable]] = OpFunction
; CHECK: OpLoopMerge %[[#]] %[[#]] DontUnroll
; CHECK-NOT: Unroll|DontUnroll
; CHECK-NOT: DontUnroll|Unroll

define dso_local void @for_loop(ptr noundef %0, i32 noundef %1) {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  store i32 0, ptr %5, align 4
  br label %6

6:                                                ; preds = %15, %2
  %7 = load i32, ptr %5, align 4
  %8 = load i32, ptr %4, align 4
  %9 = icmp slt i32 %7, %8
  br i1 %9, label %10, label %18

10:                                               ; preds = %6
  %11 = load i32, ptr %5, align 4
  %12 = load ptr, ptr %3, align 8
  %13 = load i32, ptr %12, align 4
  %14 = add nsw i32 %13, %11
  store i32 %14, ptr %12, align 4
  br label %15

15:                                               ; preds = %10
  %16 = load i32, ptr %5, align 4
  %17 = add nsw i32 %16, 1
  store i32 %17, ptr %5, align 4
  br label %6, !llvm.loop !1

18:                                               ; preds = %6
  ret void
}

define dso_local void @while_loop(ptr noundef %0, i32 noundef %1) {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  store i32 0, ptr %5, align 4
  br label %6

6:                                                ; preds = %10, %2
  %7 = load i32, ptr %5, align 4
  %8 = load i32, ptr %4, align 4
  %9 = icmp slt i32 %7, %8
  br i1 %9, label %10, label %17

10:                                               ; preds = %6
  %11 = load i32, ptr %5, align 4
  %12 = load ptr, ptr %3, align 8
  %13 = load i32, ptr %12, align 4
  %14 = add nsw i32 %13, %11
  store i32 %14, ptr %12, align 4
  %15 = load i32, ptr %5, align 4
  %16 = add nsw i32 %15, 1
  store i32 %16, ptr %5, align 4
  br label %6, !llvm.loop !3

17:                                               ; preds = %6
  ret void
}

define dso_local void @do_while_loop(ptr noundef %0, i32 noundef %1) {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  store i32 0, ptr %5, align 4
  br label %6

6:                                                ; preds = %13, %2
  %7 = load i32, ptr %5, align 4
  %8 = load ptr, ptr %3, align 8
  %9 = load i32, ptr %8, align 4
  %10 = add nsw i32 %9, %7
  store i32 %10, ptr %8, align 4
  %11 = load i32, ptr %5, align 4
  %12 = add nsw i32 %11, 1
  store i32 %12, ptr %5, align 4
  br label %13

13:                                               ; preds = %6
  %14 = load i32, ptr %5, align 4
  %15 = load i32, ptr %4, align 4
  %16 = icmp slt i32 %14, %15
  br i1 %16, label %6, label %17, !llvm.loop !4

17:                                               ; preds = %13
  ret void
}

define dso_local void @unroll_disable(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %7, %1
  %5 = load i32, ptr %3, align 4
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr %3, align 4
  br label %7

7:                                                ; preds = %4
  %8 = load i32, ptr %3, align 4
  %9 = load i32, ptr %2, align 4
  %10 = icmp slt i32 %8, %9
  br i1 %10, label %4, label %11, !llvm.loop !5

11:                                               ; preds = %7
  ret void
}

define dso_local void @unroll_count(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %7, %1
  %5 = load i32, ptr %3, align 4
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr %3, align 4
  br label %7

7:                                                ; preds = %4
  %8 = load i32, ptr %3, align 4
  %9 = load i32, ptr %2, align 4
  %10 = icmp slt i32 %8, %9
  br i1 %10, label %4, label %11, !llvm.loop !7

11:                                               ; preds = %7
  ret void
}

define dso_local void @unroll_full(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %7, %1
  %5 = load i32, ptr %3, align 4
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr %3, align 4
  br label %7

7:                                                ; preds = %4
  %8 = load i32, ptr %3, align 4
  %9 = load i32, ptr %2, align 4
  %10 = icmp slt i32 %8, %9
  br i1 %10, label %4, label %11, !llvm.loop !9

11:                                               ; preds = %7
  ret void
}

define dso_local void @unroll_full_count(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %7, %1
  %5 = load i32, ptr %3, align 4
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr %3, align 4
  br label %7

7:                                                ; preds = %4
  %8 = load i32, ptr %3, align 4
  %9 = load i32, ptr %2, align 4
  %10 = icmp slt i32 %8, %9
  br i1 %10, label %4, label %11, !llvm.loop !11

11:                                               ; preds = %7
  ret void
}

define dso_local void @unroll_enable_disable(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  store i32 0, ptr %3, align 4
  br label %4

4:                                                ; preds = %7, %1
  %5 = load i32, ptr %3, align 4
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr %3, align 4
  br label %7

7:                                                ; preds = %4
  %8 = load i32, ptr %3, align 4
  %9 = load i32, ptr %2, align 4
  %10 = icmp slt i32 %8, %9
  br i1 %10, label %4, label %11, !llvm.loop !12

11:                                               ; preds = %7
  ret void
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.enable"}
!3 = distinct !{!3, !2}
!4 = distinct !{!4, !2}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.disable"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll.count", i32 4}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.unroll.full"}
!11 = distinct !{!11, !10, !8}
!12 = distinct !{!12, !2, !6}
