; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/Large/cast.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/Large/cast.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"testVal = %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [30 x i8] c"Test %d(%s) -> %s -> %s = %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [14 x i8] c"unsigned char\00", align 1
@.str.3 = private unnamed_addr constant [12 x i8] c"signed char\00", align 1
@.str.4 = private unnamed_addr constant [15 x i8] c"unsigned short\00", align 1
@.str.5 = private unnamed_addr constant [13 x i8] c"signed short\00", align 1
@.str.6 = private unnamed_addr constant [13 x i8] c"unsigned int\00", align 1
@.str.7 = private unnamed_addr constant [11 x i8] c"signed int\00", align 1
@.str.8 = private unnamed_addr constant [19 x i8] c"unsigned long long\00", align 1
@.str.9 = private unnamed_addr constant [17 x i8] c"signed long long\00", align 1
@.str.10 = private unnamed_addr constant [21 x i8] c"%d(%s) -> bool = %d\0A\00", align 1
@.str.11 = private unnamed_addr constant [6 x i8] c"float\00", align 1
@.str.12 = private unnamed_addr constant [10 x i8] c"long long\00", align 1
@.str.13 = private unnamed_addr constant [5 x i8] c"bool\00", align 1
@.str.14 = private unnamed_addr constant [6 x i8] c"%lld\0A\00", align 1
@.str.15 = private unnamed_addr constant [4 x i8] c"%x\0A\00", align 1

; Function Attrs: mustprogress norecurse nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #4
  %8 = trunc i64 %7 to i32
  br label %9

9:                                                ; preds = %4, %2
  %10 = phi i32 [ %8, %4 ], [ 42, %2 ]
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %10)
  %12 = add nsw i32 %10, -128
  %13 = add nsw i32 %10, 127
  br label %14

14:                                               ; preds = %9, %14
  %15 = phi i32 [ %12, %9 ], [ %103, %14 ]
  %16 = and i32 %15, 255
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.2, i32 noundef %16)
  %18 = shl i32 %15, 24
  %19 = ashr exact i32 %18, 24
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.3, i32 noundef %19)
  %21 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.2, i32 noundef %16)
  %22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.3, i32 noundef %19)
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.2, i32 noundef %16)
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.3, i32 noundef %19)
  %25 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.2, i32 noundef %16)
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.3, i32 noundef %19)
  %27 = and i32 %15, 65535
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.4, i32 noundef %27)
  %29 = shl i32 %15, 16
  %30 = ashr exact i32 %29, 16
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.5, i32 noundef %30)
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.4, i32 noundef %27)
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.5, i32 noundef %30)
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.4, i32 noundef %27)
  %35 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.5, i32 noundef %30)
  %36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.4, i32 noundef %27)
  %37 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.5, i32 noundef %30)
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.6, i32 noundef %15)
  %39 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.7, i32 noundef %15)
  %40 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.6, i32 noundef %15)
  %41 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.7, i32 noundef %15)
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.6, i32 noundef %15)
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.7, i32 noundef %15)
  %44 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.6, i32 noundef %15)
  %45 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.7, i32 noundef %15)
  %46 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.8, i32 noundef %15)
  %47 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.9, i32 noundef %15)
  %48 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.8, i32 noundef %15)
  %49 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.9, i32 noundef %15)
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.8, i32 noundef %15)
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.9, i32 noundef %15)
  %52 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.8, i32 noundef %15)
  %53 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.9, i32 noundef %15)
  %54 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2, i32 noundef %16)
  %55 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3, i32 noundef %19)
  %56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.2, i32 noundef %16)
  %57 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.3, i32 noundef %19)
  %58 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2, i32 noundef %16)
  %59 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3, i32 noundef %19)
  %60 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.2, i32 noundef %16)
  %61 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.3, i32 noundef %19)
  %62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.4, i32 noundef %27)
  %63 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.5, i32 noundef %30)
  %64 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.4, i32 noundef %27)
  %65 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.5, i32 noundef %30)
  %66 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.4, i32 noundef %27)
  %67 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.5, i32 noundef %30)
  %68 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.4, i32 noundef %27)
  %69 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.5, i32 noundef %30)
  %70 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.6, i32 noundef %15)
  %71 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.7, i32 noundef %15)
  %72 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.6, i32 noundef %15)
  %73 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.7, i32 noundef %15)
  %74 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.6, i32 noundef %15)
  %75 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.7, i32 noundef %15)
  %76 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.6, i32 noundef %15)
  %77 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.7, i32 noundef %15)
  %78 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.4, i32 noundef %16)
  %79 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.5, i32 noundef %16)
  %80 = and i32 %19, 65535
  %81 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.4, i32 noundef %80)
  %82 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.5, i32 noundef %19)
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.4, i32 noundef %16)
  %84 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.5, i32 noundef %16)
  %85 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.4, i32 noundef %80)
  %86 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.5, i32 noundef %19)
  %87 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.6, i32 noundef %27)
  %88 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.7, i32 noundef %27)
  %89 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.6, i32 noundef %30)
  %90 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.7, i32 noundef %30)
  %91 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.6, i32 noundef %27)
  %92 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.7, i32 noundef %27)
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.6, i32 noundef %30)
  %94 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.5, ptr noundef nonnull @.str.7, i32 noundef %30)
  %95 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.8, i32 noundef %15)
  %96 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.9, i32 noundef %15)
  %97 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.8, i32 noundef %15)
  %98 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.9, i32 noundef %15)
  %99 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.8, i32 noundef %15)
  %100 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.9, i32 noundef %15)
  %101 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.8, i32 noundef %15)
  %102 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %15, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.9, i32 noundef %15)
  %103 = add nsw i32 %15, 1
  %104 = icmp slt i32 %15, %13
  br i1 %104, label %14, label %105, !llvm.loop !11

105:                                              ; preds = %14
  %106 = icmp ne i32 %10, 0
  %107 = zext i1 %106 to i32
  %108 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %10, ptr noundef nonnull @.str.11, i32 noundef %107)
  %109 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %10, ptr noundef nonnull @.str.12, i32 noundef %107)
  %110 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %10, ptr noundef nonnull @.str.6, i32 noundef %107)
  %111 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %10, ptr noundef nonnull @.str.3, i32 noundef %107)
  %112 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %10, ptr noundef nonnull @.str.13, i32 noundef %107)
  %113 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i64 noundef 12884901890)
  %114 = tail call noalias dereferenceable_or_null(4) ptr @malloc(i64 noundef 4) #5
  store volatile float 1.000000e+01, ptr %114, align 4, !tbaa !13
  %115 = load volatile i32, ptr %114, align 4, !tbaa !15
  %116 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i32 noundef %115)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #2

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #3

attributes #0 = { mustprogress norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { nounwind allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!14, !14, i64 0}
!14 = !{!"float", !9, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !9, i64 0}
