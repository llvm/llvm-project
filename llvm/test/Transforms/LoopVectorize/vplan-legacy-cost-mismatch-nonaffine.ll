; RUN: opt -passes=loop-vectorize -disable-output %s
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

define void @_QMmodule_cu_tiedtkePcutype(ptr noalias %0, ptr %1, ptr %2, ptr %3, i64 %4) #0 {
  br label %6

6:                                                ; preds = %24, %5
  %7 = phi i32 [ %25, %24 ], [ 0, %5 ]
  %8 = phi i64 [ %26, %24 ], [ %4, %5 ]
  %9 = icmp sgt i64 %8, 0
  br i1 %9, label %10, label %27

10:                                               ; preds = %6
  %11 = zext i32 %7 to i64
  %12 = getelementptr i32, ptr %2, i64 %11
  %13 = load i32, ptr %12, align 4
  %14 = getelementptr i32, ptr %3, i64 %11
  %15 = sext i32 %13 to i64
  %16 = mul i64 %4, %15
  %17 = getelementptr float, ptr %1, i64 %16
  %18 = load float, ptr %17, align 4
  %19 = fcmp ogt float %18, 0.000000e+00
  %20 = load i32, ptr %14, align 4
  %21 = icmp sgt i32 %20, 0
  %22 = and i1 %19, %21
  br i1 %22, label %23, label %24

23:                                               ; preds = %10
  store i32 0, ptr %0, align 4
  br label %24

24:                                               ; preds = %23, %10
  %25 = add nsw i32 %7, 1
  %26 = add i64 %8, -1
  br label %6

27:                                               ; preds = %6
  ret void
}
