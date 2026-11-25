; RUN: opt -O2 -inline-caller-loop-nest-penalty-scale=0 %s -o - \
; RUN:   | llc -mtriple=x86_64-linux-gnu -O2 \
; RUN:         -pass-remarks-missed=regalloc -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BASELINE-SSE
;
; RUN: opt -O2 %s -o - \
; RUN:   | llc -mtriple=x86_64-linux-gnu -O2 \
; RUN:         -pass-remarks-missed=regalloc -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PATCHED-SSE
;
; RUN: opt -O2 -inline-caller-loop-nest-penalty-scale=0 %s -o - \
; RUN:   | llc -mtriple=x86_64-linux-gnu -mattr=+avx2 -O2 \
; RUN:         -pass-remarks-missed=regalloc -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BASELINE-AVX2
;
; RUN: opt -O2 %s -o - \
; RUN:   | llc -mtriple=x86_64-linux-gnu -mattr=+avx2 -O2 \
; RUN:         -pass-remarks-missed=regalloc -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PATCHED-AVX2
;
; RUN: opt -O2 -inline-caller-loop-nest-penalty-scale=0 %s -o - \
; RUN:   | llc -mtriple=x86_64-linux-gnu -mattr=+avx512f -O2 \
; RUN:         -pass-remarks-missed=regalloc -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BASELINE-AVX512
;
; RUN: opt -O2 %s -o - \
; RUN:   | llc -mtriple=x86_64-linux-gnu -mattr=+avx512f -O2 \
; RUN:         -pass-remarks-missed=regalloc -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PATCHED-AVX512

; BASELINE-SSE: 76 spills
; PATCHED-SSE: 15 spills

; BASELINE-AVX2: 57 spills
; PATCHED-AVX2: 15 spills

; BASELINE-AVX512: 46 spills
; PATCHED-AVX512: 15 spills

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @vec_update_unrolled(i32 noundef %0, double noundef nofpclass(nan inf) %1, ptr noundef %2, i32 noundef %3, ptr noundef %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = icmp slt i32 %0, 1
  %8 = fcmp fast oeq double %1, 0.000000e+00
  %9 = select i1 %7, i1 true, i1 %8
  br i1 %9, label %64, label %10

10:                                               ; preds = %6
  %11 = and i32 %0, 3
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %27, label %13

13:                                               ; preds = %10, %16
  %14 = phi i32 [ %24, %16 ], [ 0, %10 ]
  %15 = icmp samesign ult i32 %14, %11
  br i1 %15, label %16, label %25

16:                                               ; preds = %13
  %17 = zext nneg i32 %14 to i64
  %18 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %17
  %19 = load double, ptr %18, align 8
  %20 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %17
  %21 = load double, ptr %20, align 8
  %22 = fmul fast double %1, %21
  %23 = fadd fast double %19, %22
  store double %23, ptr %18, align 8
  %24 = add nuw nsw i32 %14, 1
  br label %13

25:                                               ; preds = %13
  %26 = icmp slt i32 %0, 4
  br i1 %26, label %64, label %27

27:                                               ; preds = %25, %10
  br label %28

28:                                               ; preds = %31, %27
  %29 = phi i32 [ %11, %27 ], [ %63, %31 ]
  %30 = icmp slt i32 %29, %0
  br i1 %30, label %31, label %64

31:                                               ; preds = %28
  %32 = zext nneg i32 %29 to i64
  %33 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %32
  %34 = load double, ptr %33, align 8
  %35 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %32
  %36 = load double, ptr %35, align 8
  %37 = fmul fast double %1, %36
  %38 = fadd fast double %34, %37
  store double %38, ptr %33, align 8
  %39 = add nuw nsw i32 %29, 1
  %40 = zext nneg i32 %39 to i64
  %41 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %40
  %42 = load double, ptr %41, align 8
  %43 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %40
  %44 = load double, ptr %43, align 8
  %45 = fmul fast double %1, %44
  %46 = fadd fast double %42, %45
  store double %46, ptr %41, align 8
  %47 = add nuw nsw i32 %29, 2
  %48 = zext nneg i32 %47 to i64
  %49 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %48
  %50 = load double, ptr %49, align 8
  %51 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %48
  %52 = load double, ptr %51, align 8
  %53 = fmul fast double %1, %52
  %54 = fadd fast double %50, %53
  store double %54, ptr %49, align 8
  %55 = add nuw nsw i32 %29, 3
  %56 = zext nneg i32 %55 to i64
  %57 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %56
  %58 = load double, ptr %57, align 8
  %59 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %56
  %60 = load double, ptr %59, align 8
  %61 = fmul fast double %1, %60
  %62 = fadd fast double %58, %61
  store double %62, ptr %57, align 8
  %63 = add nuw nsw i32 %29, 4
  br label %28

64:                                               ; preds = %28, %25, %6
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @matrix_factor(ptr noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %3, ptr noundef %4) local_unnamed_addr #1 {
  store i32 0, ptr %4, align 4
  %6 = add nsw i32 %2, -1
  %7 = icmp sgt i32 %2, 1
  br i1 %7, label %8, label %51

8:                                                ; preds = %5, %50
  %9 = phi i32 [ %12, %50 ], [ 0, %5 ]
  %10 = icmp slt i32 %9, %6
  br i1 %10, label %11, label %51

11:                                               ; preds = %8
  %12 = add nuw nsw i32 %9, 1
  %13 = zext nneg i32 %9 to i64
  %14 = getelementptr inbounds nuw [4 x i8], ptr %3, i64 %13
  store i32 %9, ptr %14, align 4
  %15 = mul nsw i32 %1, %9
  %16 = add nsw i32 %15, %9
  %17 = sext i32 %16 to i64
  %18 = getelementptr inbounds [8 x i8], ptr %0, i64 %17
  %19 = load double, ptr %18, align 8
  %20 = fcmp fast oeq double %19, 0.000000e+00
  br i1 %20, label %21, label %22

21:                                               ; preds = %11
  store i32 %9, ptr %4, align 4
  br label %50

22:                                               ; preds = %11, %25
  %23 = phi i32 [ %32, %25 ], [ %12, %11 ]
  %24 = icmp slt i32 %23, %2
  br i1 %24, label %25, label %33

25:                                               ; preds = %22
  %26 = add nsw i32 %15, %23
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds [8 x i8], ptr %0, i64 %27
  %29 = load double, ptr %28, align 8
  %30 = fneg fast double %29
  %31 = fdiv fast double %30, %19
  store double %31, ptr %28, align 8
  %32 = add nuw nsw i32 %23, 1
  br label %22

33:                                               ; preds = %22, %36
  %34 = phi i32 [ %49, %36 ], [ %12, %22 ]
  %35 = icmp slt i32 %34, %2
  br i1 %35, label %36, label %50

36:                                               ; preds = %33
  %37 = mul nsw i32 %1, %34
  %38 = add nsw i32 %37, %9
  %39 = sext i32 %38 to i64
  %40 = getelementptr inbounds [8 x i8], ptr %0, i64 %39
  %41 = load double, ptr %40, align 8
  %42 = sub nsw i32 %2, %12
  %43 = add nsw i32 %15, %12
  %44 = sext i32 %43 to i64
  %45 = getelementptr inbounds [8 x i8], ptr %0, i64 %44
  %46 = add nsw i32 %37, %12
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds [8 x i8], ptr %0, i64 %47
  call void @vec_update_unrolled(i32 noundef %42, double noundef nofpclass(nan inf) %41, ptr noundef %45, i32 noundef 1, ptr noundef %48, i32 noundef 1)
  %49 = add nuw nsw i32 %34, 1
  br label %33

50:                                               ; preds = %33, %21
  br label %8

51:                                               ; preds = %8, %5
  %52 = sext i32 %6 to i64
  %53 = getelementptr inbounds [4 x i8], ptr %3, i64 %52
  store i32 %6, ptr %53, align 4
  %54 = add i32 %1, 1
  %55 = mul i32 %54, %6
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds [8 x i8], ptr %0, i64 %56
  %58 = load double, ptr %57, align 8
  %59 = fcmp fast oeq double %58, 0.000000e+00
  br i1 %59, label %60, label %61

60:                                               ; preds = %51
  store i32 %6, ptr %4, align 4
  br label %61

61:                                               ; preds = %60, %51
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @triangular_solve(ptr noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %3, ptr noundef %4) local_unnamed_addr #1 {
  %6 = add nsw i32 %2, -1
  br label %7

7:                                                ; preds = %21, %5
  %8 = phi i32 [ 0, %5 ], [ %22, %21 ]
  %9 = icmp slt i32 %8, %6
  br i1 %9, label %10, label %31

10:                                               ; preds = %7
  %11 = zext nneg i32 %8 to i64
  %12 = getelementptr inbounds nuw [4 x i8], ptr %3, i64 %11
  %13 = load i32, ptr %12, align 4
  %14 = sext i32 %13 to i64
  %15 = getelementptr inbounds [8 x i8], ptr %4, i64 %14
  %16 = load double, ptr %15, align 8
  %17 = icmp eq i32 %13, %8
  br i1 %17, label %21, label %18

18:                                               ; preds = %10
  %19 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %11
  %20 = load double, ptr %19, align 8
  store double %20, ptr %15, align 8
  store double %16, ptr %19, align 8
  br label %21

21:                                               ; preds = %18, %10
  %22 = add nuw nsw i32 %8, 1
  %23 = sub nsw i32 %2, %22
  %24 = add i32 %1, 1
  %25 = mul i32 %24, %8
  %26 = sext i32 %25 to i64
  %27 = getelementptr [8 x i8], ptr %0, i64 %26
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = zext nneg i32 %22 to i64
  %30 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %29
  call void @vec_update_unrolled(i32 noundef %23, double noundef nofpclass(nan inf) %16, ptr noundef %28, i32 noundef 1, ptr noundef nonnull %30, i32 noundef 1)
  br label %7

31:                                               ; preds = %7, %34
  %32 = phi i32 [ %46, %34 ], [ %6, %7 ]
  %33 = icmp sgt i32 %32, -1
  br i1 %33, label %34, label %47

34:                                               ; preds = %31
  %35 = zext nneg i32 %32 to i64
  %36 = getelementptr inbounds nuw [8 x i8], ptr %4, i64 %35
  %37 = load double, ptr %36, align 8
  %38 = mul nsw i32 %1, %32
  %39 = add nsw i32 %38, %32
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds [8 x i8], ptr %0, i64 %40
  %42 = load double, ptr %41, align 8
  %43 = fdiv fast double %37, %42
  store double %43, ptr %36, align 8
  %44 = sext i32 %38 to i64
  %45 = getelementptr inbounds [8 x i8], ptr %0, i64 %44
  call void @vec_update_unrolled(i32 noundef %32, double noundef nofpclass(nan inf) %43, ptr noundef %45, i32 noundef 1, ptr noundef %4, i32 noundef 1)
  %46 = add nsw i32 %32, -1
  br label %31

47:                                               ; preds = %31
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind uwtable }
