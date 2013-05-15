; ModuleID = 'matmul.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [1536 x [1536 x float]] zeroinitializer, align 16
@B = common global [1536 x [1536 x float]] zeroinitializer, align 16
@stdout = external global %struct._IO_FILE*
@.str = private unnamed_addr constant [5 x i8] c"%lf \00"
@C = common global [1536 x [1536 x float]] zeroinitializer, align 16
@.str1 = private unnamed_addr constant [2 x i8] c"\0A\00"

define void @init_array() nounwind {
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, i32* %i, align 4
  br label %1

; <label>:1                                       ; preds = %41, %0
  %2 = load i32* %i, align 4
  %3 = icmp slt i32 %2, 1536
  br i1 %3, label %4, label %44

; <label>:4                                       ; preds = %1
  store i32 0, i32* %j, align 4
  br label %5

; <label>:5                                       ; preds = %37, %4
  %6 = load i32* %j, align 4
  %7 = icmp slt i32 %6, 1536
  br i1 %7, label %8, label %40

; <label>:8                                       ; preds = %5
  %9 = load i32* %i, align 4
  %10 = load i32* %j, align 4
  %11 = mul nsw i32 %9, %10
  %12 = srem i32 %11, 1024
  %13 = add nsw i32 1, %12
  %14 = sitofp i32 %13 to double
  %15 = fdiv double %14, 2.000000e+00
  %16 = fptrunc double %15 to float
  %17 = load i32* %j, align 4
  %18 = sext i32 %17 to i64
  %19 = load i32* %i, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds [1536 x [1536 x float]]* @A, i32 0, i64 %20
  %22 = getelementptr inbounds [1536 x float]* %21, i32 0, i64 %18
  store float %16, float* %22
  %23 = load i32* %i, align 4
  %24 = load i32* %j, align 4
  %25 = mul nsw i32 %23, %24
  %26 = srem i32 %25, 1024
  %27 = add nsw i32 1, %26
  %28 = sitofp i32 %27 to double
  %29 = fdiv double %28, 2.000000e+00
  %30 = fptrunc double %29 to float
  %31 = load i32* %j, align 4
  %32 = sext i32 %31 to i64
  %33 = load i32* %i, align 4
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds [1536 x [1536 x float]]* @B, i32 0, i64 %34
  %36 = getelementptr inbounds [1536 x float]* %35, i32 0, i64 %32
  store float %30, float* %36
  br label %37

; <label>:37                                      ; preds = %8
  %38 = load i32* %j, align 4
  %39 = add nsw i32 %38, 1
  store i32 %39, i32* %j, align 4
  br label %5

; <label>:40                                      ; preds = %5
  br label %41

; <label>:41                                      ; preds = %40
  %42 = load i32* %i, align 4
  %43 = add nsw i32 %42, 1
  store i32 %43, i32* %i, align 4
  br label %1

; <label>:44                                      ; preds = %1
  ret void
}

define void @print_array() nounwind {
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, i32* %i, align 4
  br label %1

; <label>:1                                       ; preds = %32, %0
  %2 = load i32* %i, align 4
  %3 = icmp slt i32 %2, 1536
  br i1 %3, label %4, label %35

; <label>:4                                       ; preds = %1
  store i32 0, i32* %j, align 4
  br label %5

; <label>:5                                       ; preds = %26, %4
  %6 = load i32* %j, align 4
  %7 = icmp slt i32 %6, 1536
  br i1 %7, label %8, label %29

; <label>:8                                       ; preds = %5
  %9 = load %struct._IO_FILE** @stdout, align 8
  %10 = load i32* %j, align 4
  %11 = sext i32 %10 to i64
  %12 = load i32* %i, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds [1536 x [1536 x float]]* @C, i32 0, i64 %13
  %15 = getelementptr inbounds [1536 x float]* %14, i32 0, i64 %11
  %16 = load float* %15
  %17 = fpext float %16 to double
  %18 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %9, i8* getelementptr inbounds ([5 x i8]* @.str, i32 0, i32 0), double %17)
  %19 = load i32* %j, align 4
  %20 = srem i32 %19, 80
  %21 = icmp eq i32 %20, 79
  br i1 %21, label %22, label %25

; <label>:22                                      ; preds = %8
  %23 = load %struct._IO_FILE** @stdout, align 8
  %24 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %23, i8* getelementptr inbounds ([2 x i8]* @.str1, i32 0, i32 0))
  br label %25

; <label>:25                                      ; preds = %22, %8
  br label %26

; <label>:26                                      ; preds = %25
  %27 = load i32* %j, align 4
  %28 = add nsw i32 %27, 1
  store i32 %28, i32* %j, align 4
  br label %5

; <label>:29                                      ; preds = %5
  %30 = load %struct._IO_FILE** @stdout, align 8
  %31 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %30, i8* getelementptr inbounds ([2 x i8]* @.str1, i32 0, i32 0))
  br label %32

; <label>:32                                      ; preds = %29
  %33 = load i32* %i, align 4
  %34 = add nsw i32 %33, 1
  store i32 %34, i32* %i, align 4
  br label %1

; <label>:35                                      ; preds = %1
  ret void
}

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...)

define i32 @main() nounwind {
  %1 = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %t_start = alloca double, align 8
  %t_end = alloca double, align 8
  store i32 0, i32* %1
  call void @init_array()
  store i32 0, i32* %i, align 4
  br label %2

; <label>:2                                       ; preds = %57, %0
  %3 = load i32* %i, align 4
  %4 = icmp slt i32 %3, 1536
  br i1 %4, label %5, label %60

; <label>:5                                       ; preds = %2
  store i32 0, i32* %j, align 4
  br label %6

; <label>:6                                       ; preds = %53, %5
  %7 = load i32* %j, align 4
  %8 = icmp slt i32 %7, 1536
  br i1 %8, label %9, label %56

; <label>:9                                       ; preds = %6
  %10 = load i32* %j, align 4
  %11 = sext i32 %10 to i64
  %12 = load i32* %i, align 4
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds [1536 x [1536 x float]]* @C, i32 0, i64 %13
  %15 = getelementptr inbounds [1536 x float]* %14, i32 0, i64 %11
  store float 0.000000e+00, float* %15
  store i32 0, i32* %k, align 4
  br label %16

; <label>:16                                      ; preds = %49, %9
  %17 = load i32* %k, align 4
  %18 = icmp slt i32 %17, 1536
  br i1 %18, label %19, label %52

; <label>:19                                      ; preds = %16
  %20 = load i32* %j, align 4
  %21 = sext i32 %20 to i64
  %22 = load i32* %i, align 4
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds [1536 x [1536 x float]]* @C, i32 0, i64 %23
  %25 = getelementptr inbounds [1536 x float]* %24, i32 0, i64 %21
  %26 = load float* %25
  %27 = load i32* %k, align 4
  %28 = sext i32 %27 to i64
  %29 = load i32* %i, align 4
  %30 = sext i32 %29 to i64
  %31 = getelementptr inbounds [1536 x [1536 x float]]* @A, i32 0, i64 %30
  %32 = getelementptr inbounds [1536 x float]* %31, i32 0, i64 %28
  %33 = load float* %32
  %34 = load i32* %j, align 4
  %35 = sext i32 %34 to i64
  %36 = load i32* %k, align 4
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds [1536 x [1536 x float]]* @B, i32 0, i64 %37
  %39 = getelementptr inbounds [1536 x float]* %38, i32 0, i64 %35
  %40 = load float* %39
  %41 = fmul float %33, %40
  %42 = fadd float %26, %41
  %43 = load i32* %j, align 4
  %44 = sext i32 %43 to i64
  %45 = load i32* %i, align 4
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds [1536 x [1536 x float]]* @C, i32 0, i64 %46
  %48 = getelementptr inbounds [1536 x float]* %47, i32 0, i64 %44
  store float %42, float* %48
  br label %49

; <label>:49                                      ; preds = %19
  %50 = load i32* %k, align 4
  %51 = add nsw i32 %50, 1
  store i32 %51, i32* %k, align 4
  br label %16

; <label>:52                                      ; preds = %16
  br label %53

; <label>:53                                      ; preds = %52
  %54 = load i32* %j, align 4
  %55 = add nsw i32 %54, 1
  store i32 %55, i32* %j, align 4
  br label %6

; <label>:56                                      ; preds = %6
  br label %57

; <label>:57                                      ; preds = %56
  %58 = load i32* %i, align 4
  %59 = add nsw i32 %58, 1
  store i32 %59, i32* %i, align 4
  br label %2

; <label>:60                                      ; preds = %2
  ret i32 0
}
