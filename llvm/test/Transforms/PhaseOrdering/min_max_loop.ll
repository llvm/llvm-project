; RUN: opt < %s --O3 -S | FileCheck %s

define signext i16 @vecreduce_smax_v2i16(i32 noundef %0, ptr noundef %1) #0 {
   ;; CHECK: @llvm.smax
   %3 = alloca i32, align 4
   %4 = alloca ptr, align 8
   %5 = alloca i16, align 2
   %6 = alloca i32, align 4
   store i32 %0, ptr %3, align 4
   store ptr %1, ptr %4, align 8
   store i16 0, ptr %5, align 2
   store i32 0, ptr %6, align 4
   br label %7
 
 7:                                                ; preds = %34, %2
   %8 = load i32, ptr %6, align 4
   %9 = load i32, ptr %3, align 4
   %10 = icmp slt i32 %8, %9
   br i1 %10, label %11, label %37
 
 11:                                               ; preds = %7
   %12 = load i16, ptr %5, align 2
   %13 = sext i16 %12 to i32
   %14 = load ptr, ptr %4, align 8
   %15 = load i32, ptr %6, align 4
   %16 = sext i32 %15 to i64
   %17 = getelementptr inbounds i16, ptr %14, i64 %16
   %18 = load i16, ptr %17, align 2
   %19 = sext i16 %18 to i32
   %20 = icmp slt i32 %13, %19
   br i1 %20, label %21, label %28
 
 21:                                               ; preds = %11
   %22 = load ptr, ptr %4, align 8
   %23 = load i32, ptr %6, align 4
   %24 = sext i32 %23 to i64
   %25 = getelementptr inbounds i16, ptr %22, i64 %24
   %26 = load i16, ptr %25, align 2
   %27 = sext i16 %26 to i32
   br label %31
 
 28:                                               ; preds = %11
   %29 = load i16, ptr %5, align 2
   %30 = sext i16 %29 to i32
   br label %31
 
 31:                                               ; preds = %28, %21
   %32 = phi i32 [ %27, %21 ], [ %30, %28 ]
   %33 = trunc i32 %32 to i16
   store i16 %33, ptr %5, align 2
   br label %34
 
 34:                                               ; preds = %31
   %35 = load i32, ptr %6, align 4
   %36 = add nsw i32 %35, 1
   store i32 %36, ptr %6, align 4
   br label %7
 
 37:                                               ; preds = %7
   %38 = load i16, ptr %5, align 2
   ret i16 %38
 }

define signext i16 @vecreduce_smin_v2i16(i32 noundef %0, ptr noundef %1) #0 {
; CHECK: @llvm.smin
  %3 = alloca i32, align 4
  %4 = alloca ptr, align 8
  %5 = alloca i16, align 2
  %6 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store ptr %1, ptr %4, align 8
  store i16 0, ptr %5, align 2
  store i32 0, ptr %6, align 4
  br label %7

7:                                                ; preds = %34, %2
  %8 = load i32, ptr %6, align 4
  %9 = load i32, ptr %3, align 4
  %10 = icmp slt i32 %8, %9
  br i1 %10, label %11, label %37

11:                                               ; preds = %7
  %12 = load i16, ptr %5, align 2
  %13 = sext i16 %12 to i32
  %14 = load ptr, ptr %4, align 8
  %15 = load i32, ptr %6, align 4
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds i16, ptr %14, i64 %16
  %18 = load i16, ptr %17, align 2
  %19 = sext i16 %18 to i32
  %20 = icmp sgt i32 %13, %19
  br i1 %20, label %21, label %28

21:                                               ; preds = %11
  %22 = load ptr, ptr %4, align 8
  %23 = load i32, ptr %6, align 4
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds i16, ptr %22, i64 %24
  %26 = load i16, ptr %25, align 2
  %27 = sext i16 %26 to i32
  br label %31

28:                                               ; preds = %11
  %29 = load i16, ptr %5, align 2
  %30 = sext i16 %29 to i32
  br label %31

31:                                               ; preds = %28, %21
  %32 = phi i32 [ %27, %21 ], [ %30, %28 ]
  %33 = trunc i32 %32 to i16
  store i16 %33, ptr %5, align 2
  br label %34

34:                                               ; preds = %31
  %35 = load i32, ptr %6, align 4
  %36 = add nsw i32 %35, 1
  store i32 %36, ptr %6, align 4
  br label %7 

37:                                               ; preds = %7
  %38 = load i16, ptr %5, align 2
  ret i16 %38
}



