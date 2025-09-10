; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/puzzle.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/puzzle.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@next = internal unnamed_addr global i64 1, align 8
@.str = private unnamed_addr constant [21 x i8] c"Found duplicate: %d\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 1, 32768) i32 @rand() local_unnamed_addr #0 {
  %1 = load i64, ptr @next, align 8, !tbaa !6
  %2 = mul i64 %1, 1103515245
  %3 = add i64 %2, 12345
  store i64 %3, ptr @next, align 8, !tbaa !6
  %4 = lshr i64 %3, 16
  %5 = trunc i64 %4 to i32
  %6 = urem i32 %5, 32767
  %7 = add nuw nsw i32 %6, 1
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @srand(i32 noundef %0) local_unnamed_addr #1 {
  %2 = zext i32 %0 to i64
  store i64 %2, ptr @next, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local i32 @randInt(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = sub nsw i32 %1, %0
  %4 = add nsw i32 %3, 1
  %5 = sitofp i32 %4 to double
  %6 = load i64, ptr @next, align 8, !tbaa !6
  %7 = mul i64 %6, 1103515245
  %8 = add i64 %7, 12345
  store i64 %8, ptr @next, align 8, !tbaa !6
  %9 = lshr i64 %8, 16
  %10 = trunc i64 %9 to i32
  %11 = urem i32 %10, 32767
  %12 = add nuw nsw i32 %11, 1
  %13 = uitofp nneg i32 %12 to double
  %14 = fmul double %13, 0x3F00000000000000
  %15 = fmul double %14, %5
  %16 = fptosi double %15 to i32
  %17 = icmp eq i32 %4, %16
  %18 = add nsw i32 %0, %16
  %19 = sext i1 %17 to i32
  %20 = add nsw i32 %18, %19
  ret i32 %20
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @shuffle(ptr noundef captures(none) %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = add nsw i32 %1, -1
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %34, label %5

5:                                                ; preds = %2
  %6 = load i64, ptr @next, align 8
  %7 = sext i32 %3 to i64
  br label %8

8:                                                ; preds = %5, %8
  %9 = phi i64 [ %31, %8 ], [ %7, %5 ]
  %10 = phi i64 [ %14, %8 ], [ %6, %5 ]
  %11 = add i64 %9, 1
  %12 = uitofp i64 %11 to double
  %13 = mul i64 %10, 1103515245
  %14 = add i64 %13, 12345
  %15 = lshr i64 %14, 16
  %16 = trunc i64 %15 to i32
  %17 = urem i32 %16, 32767
  %18 = add nuw nsw i32 %17, 1
  %19 = uitofp nneg i32 %18 to double
  %20 = fmul double %19, 0x3F00000000000000
  %21 = fmul double %20, %12
  %22 = fptosi double %21 to i32
  %23 = sext i32 %22 to i64
  %24 = icmp eq i64 %11, %23
  %25 = sext i1 %24 to i64
  %26 = getelementptr inbounds nuw i32, ptr %0, i64 %9
  %27 = load i32, ptr %26, align 4, !tbaa !10
  %28 = getelementptr i32, ptr %0, i64 %23
  %29 = getelementptr i32, ptr %28, i64 %25
  %30 = load i32, ptr %29, align 4, !tbaa !10
  store i32 %30, ptr %26, align 4, !tbaa !10
  store i32 %27, ptr %29, align 4, !tbaa !10
  %31 = add i64 %9, -1
  %32 = icmp eq i64 %31, 0
  br i1 %32, label %33, label %8, !llvm.loop !12

33:                                               ; preds = %8
  store i64 %14, ptr @next, align 8, !tbaa !6
  br label %34

34:                                               ; preds = %33, %2
  ret void
}

; Function Attrs: nofree nounwind memory(readwrite, argmem: none) uwtable
define dso_local noalias noundef ptr @createRandomArray(i32 noundef %0) local_unnamed_addr #3 {
  %2 = add i32 %0, 1
  %3 = sext i32 %2 to i64
  %4 = shl nsw i64 %3, 2
  %5 = tail call noalias ptr @malloc(i64 noundef %4) #10
  %6 = icmp slt i32 %0, 0
  br i1 %6, label %31, label %7

7:                                                ; preds = %1
  %8 = zext i32 %2 to i64
  %9 = icmp ult i32 %2, 8
  br i1 %9, label %23, label %10

10:                                               ; preds = %7
  %11 = and i64 %8, 4294967288
  br label %12

12:                                               ; preds = %12, %10
  %13 = phi i64 [ 0, %10 ], [ %18, %12 ]
  %14 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %10 ], [ %19, %12 ]
  %15 = add <4 x i32> %14, splat (i32 4)
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %13
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 16
  store <4 x i32> %14, ptr %16, align 4, !tbaa !10
  store <4 x i32> %15, ptr %17, align 4, !tbaa !10
  %18 = add nuw i64 %13, 8
  %19 = add <4 x i32> %14, splat (i32 8)
  %20 = icmp eq i64 %18, %11
  br i1 %20, label %21, label %12, !llvm.loop !14

21:                                               ; preds = %12
  %22 = icmp eq i64 %11, %8
  br i1 %22, label %31, label %23

23:                                               ; preds = %7, %21
  %24 = phi i64 [ 0, %7 ], [ %11, %21 ]
  br label %25

25:                                               ; preds = %23, %25
  %26 = phi i64 [ %29, %25 ], [ %24, %23 ]
  %27 = getelementptr inbounds nuw i32, ptr %5, i64 %26
  %28 = trunc nuw nsw i64 %26 to i32
  store i32 %28, ptr %27, align 4, !tbaa !10
  %29 = add nuw nsw i64 %26, 1
  %30 = icmp eq i64 %29, %8
  br i1 %30, label %31, label %25, !llvm.loop !17

31:                                               ; preds = %25, %21, %1
  %32 = sitofp i32 %0 to double
  %33 = load i64, ptr @next, align 8, !tbaa !6
  %34 = mul i64 %33, 1103515245
  %35 = add i64 %34, 12345
  store i64 %35, ptr @next, align 8, !tbaa !6
  %36 = lshr i64 %35, 16
  %37 = trunc i64 %36 to i32
  %38 = urem i32 %37, 32767
  %39 = add nuw nsw i32 %38, 1
  %40 = uitofp nneg i32 %39 to double
  %41 = fmul double %40, 0x3F00000000000000
  %42 = fmul double %41, %32
  %43 = fptosi double %42 to i32
  %44 = icmp eq i32 %0, %43
  %45 = add nsw i32 %43, 1
  %46 = sext i1 %44 to i32
  %47 = add nsw i32 %45, %46
  store i32 %47, ptr %5, align 4, !tbaa !10
  %48 = icmp eq i32 %0, 0
  br i1 %48, label %77, label %49

49:                                               ; preds = %31
  %50 = sext i32 %0 to i64
  br label %51

51:                                               ; preds = %51, %49
  %52 = phi i64 [ %74, %51 ], [ %50, %49 ]
  %53 = phi i64 [ %57, %51 ], [ %35, %49 ]
  %54 = add i64 %52, 1
  %55 = uitofp i64 %54 to double
  %56 = mul i64 %53, 1103515245
  %57 = add i64 %56, 12345
  %58 = lshr i64 %57, 16
  %59 = trunc i64 %58 to i32
  %60 = urem i32 %59, 32767
  %61 = add nuw nsw i32 %60, 1
  %62 = uitofp nneg i32 %61 to double
  %63 = fmul double %62, 0x3F00000000000000
  %64 = fmul double %63, %55
  %65 = fptosi double %64 to i32
  %66 = sext i32 %65 to i64
  %67 = icmp eq i64 %54, %66
  %68 = sext i1 %67 to i64
  %69 = getelementptr inbounds nuw i32, ptr %5, i64 %52
  %70 = load i32, ptr %69, align 4, !tbaa !10
  %71 = getelementptr i32, ptr %5, i64 %66
  %72 = getelementptr i32, ptr %71, i64 %68
  %73 = load i32, ptr %72, align 4, !tbaa !10
  store i32 %73, ptr %69, align 4, !tbaa !10
  store i32 %70, ptr %72, align 4, !tbaa !10
  %74 = add i64 %52, -1
  %75 = icmp eq i64 %74, 0
  br i1 %75, label %76, label %51, !llvm.loop !12

76:                                               ; preds = %51
  store i64 %57, ptr @next, align 8, !tbaa !6
  br label %77

77:                                               ; preds = %31, %76
  ret ptr %5
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #4

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local i32 @findDuplicate(ptr noundef readonly captures(none) %0, i32 noundef %1) local_unnamed_addr #5 {
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %46

4:                                                ; preds = %2
  %5 = zext nneg i32 %1 to i64
  %6 = icmp ult i32 %1, 8
  br i1 %6, label %33, label %7

7:                                                ; preds = %4
  %8 = and i64 %5, 2147483640
  br label %9

9:                                                ; preds = %9, %7
  %10 = phi i64 [ 0, %7 ], [ %26, %9 ]
  %11 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %7 ], [ %27, %9 ]
  %12 = phi <4 x i32> [ zeroinitializer, %7 ], [ %24, %9 ]
  %13 = phi <4 x i32> [ zeroinitializer, %7 ], [ %25, %9 ]
  %14 = getelementptr inbounds nuw i32, ptr %0, i64 %10
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %16 = load <4 x i32>, ptr %14, align 4, !tbaa !10
  %17 = load <4 x i32>, ptr %15, align 4, !tbaa !10
  %18 = xor <4 x i32> %12, %16
  %19 = xor <4 x i32> %13, %17
  %20 = trunc <4 x i64> %11 to <4 x i32>
  %21 = add <4 x i32> %20, splat (i32 1)
  %22 = trunc <4 x i64> %11 to <4 x i32>
  %23 = add <4 x i32> %22, splat (i32 5)
  %24 = xor <4 x i32> %18, %21
  %25 = xor <4 x i32> %19, %23
  %26 = add nuw i64 %10, 8
  %27 = add <4 x i64> %11, splat (i64 8)
  %28 = icmp eq i64 %26, %8
  br i1 %28, label %29, label %9, !llvm.loop !18

29:                                               ; preds = %9
  %30 = xor <4 x i32> %25, %24
  %31 = tail call i32 @llvm.vector.reduce.xor.v4i32(<4 x i32> %30)
  %32 = icmp eq i64 %8, %5
  br i1 %32, label %46, label %33

33:                                               ; preds = %4, %29
  %34 = phi i64 [ 0, %4 ], [ %8, %29 ]
  %35 = phi i32 [ 0, %4 ], [ %31, %29 ]
  br label %36

36:                                               ; preds = %33, %36
  %37 = phi i64 [ %39, %36 ], [ %34, %33 ]
  %38 = phi i32 [ %44, %36 ], [ %35, %33 ]
  %39 = add nuw nsw i64 %37, 1
  %40 = getelementptr inbounds nuw i32, ptr %0, i64 %37
  %41 = load i32, ptr %40, align 4, !tbaa !10
  %42 = xor i32 %38, %41
  %43 = trunc nuw nsw i64 %39 to i32
  %44 = xor i32 %42, %43
  %45 = icmp eq i64 %39, %5
  br i1 %45, label %46, label %36, !llvm.loop !19

46:                                               ; preds = %36, %29, %2
  %47 = phi i32 [ 0, %2 ], [ %31, %29 ], [ %44, %36 ]
  %48 = xor i32 %47, %1
  ret i32 %48
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 {
  store i64 1, ptr @next, align 8, !tbaa !6
  br label %1

1:                                                ; preds = %0, %82
  %2 = phi i32 [ 0, %0 ], [ %88, %82 ]
  %3 = tail call noalias dereferenceable_or_null(2000004) ptr @malloc(i64 noundef 2000004) #10
  br label %4

4:                                                ; preds = %4, %1
  %5 = phi i64 [ 0, %1 ], [ %10, %4 ]
  %6 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %1 ], [ %11, %4 ]
  %7 = add <4 x i32> %6, splat (i32 4)
  %8 = getelementptr inbounds nuw i32, ptr %3, i64 %5
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store <4 x i32> %6, ptr %8, align 4, !tbaa !10
  store <4 x i32> %7, ptr %9, align 4, !tbaa !10
  %10 = add nuw i64 %5, 8
  %11 = add <4 x i32> %6, splat (i32 8)
  %12 = icmp eq i64 %10, 500000
  br i1 %12, label %13, label %4, !llvm.loop !20

13:                                               ; preds = %4
  %14 = getelementptr inbounds nuw i8, ptr %3, i64 2000000
  store i32 500000, ptr %14, align 4, !tbaa !10
  %15 = load i64, ptr @next, align 8, !tbaa !6
  %16 = mul i64 %15, 1103515245
  %17 = add i64 %16, 12345
  %18 = lshr i64 %17, 16
  %19 = trunc i64 %18 to i32
  %20 = urem i32 %19, 32767
  %21 = add nuw nsw i32 %20, 1
  %22 = uitofp nneg i32 %21 to double
  %23 = fmul double %22, 0x3F00000000000000
  %24 = fmul double %23, 5.000000e+05
  %25 = fptosi double %24 to i32
  %26 = icmp eq i32 %25, 500000
  %27 = add nsw i32 %25, 1
  %28 = sext i1 %26 to i32
  %29 = add nsw i32 %27, %28
  store i32 %29, ptr %3, align 4, !tbaa !10
  br label %30

30:                                               ; preds = %30, %13
  %31 = phi i64 [ %53, %30 ], [ 500000, %13 ]
  %32 = phi i64 [ %36, %30 ], [ %17, %13 ]
  %33 = add nuw nsw i64 %31, 1
  %34 = uitofp i64 %33 to double
  %35 = mul i64 %32, 1103515245
  %36 = add i64 %35, 12345
  %37 = lshr i64 %36, 16
  %38 = trunc i64 %37 to i32
  %39 = urem i32 %38, 32767
  %40 = add nuw nsw i32 %39, 1
  %41 = uitofp nneg i32 %40 to double
  %42 = fmul double %41, 0x3F00000000000000
  %43 = fmul double %42, %34
  %44 = fptosi double %43 to i32
  %45 = sext i32 %44 to i64
  %46 = icmp eq i64 %33, %45
  %47 = sext i1 %46 to i64
  %48 = getelementptr inbounds nuw i32, ptr %3, i64 %31
  %49 = load i32, ptr %48, align 4, !tbaa !10
  %50 = getelementptr i32, ptr %3, i64 %45
  %51 = getelementptr i32, ptr %50, i64 %47
  %52 = load i32, ptr %51, align 4, !tbaa !10
  store i32 %52, ptr %48, align 4, !tbaa !10
  store i32 %49, ptr %51, align 4, !tbaa !10
  %53 = add nsw i64 %31, -1
  %54 = icmp eq i64 %53, 0
  br i1 %54, label %55, label %30, !llvm.loop !12

55:                                               ; preds = %30
  store i64 %36, ptr @next, align 8, !tbaa !6
  %56 = getelementptr inbounds nuw i8, ptr %3, i64 2000000
  br label %57

57:                                               ; preds = %79, %55
  %58 = phi i32 [ 0, %55 ], [ %80, %79 ]
  br label %59

59:                                               ; preds = %59, %57
  %60 = phi i64 [ 0, %57 ], [ %76, %59 ]
  %61 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %57 ], [ %77, %59 ]
  %62 = phi <4 x i32> [ zeroinitializer, %57 ], [ %74, %59 ]
  %63 = phi <4 x i32> [ zeroinitializer, %57 ], [ %75, %59 ]
  %64 = getelementptr inbounds nuw i32, ptr %3, i64 %60
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 16
  %66 = load <4 x i32>, ptr %64, align 4, !tbaa !10
  %67 = load <4 x i32>, ptr %65, align 4, !tbaa !10
  %68 = xor <4 x i32> %66, %62
  %69 = xor <4 x i32> %67, %63
  %70 = trunc <4 x i64> %61 to <4 x i32>
  %71 = add <4 x i32> %70, splat (i32 1)
  %72 = trunc <4 x i64> %61 to <4 x i32>
  %73 = add <4 x i32> %72, splat (i32 5)
  %74 = xor <4 x i32> %68, %71
  %75 = xor <4 x i32> %69, %73
  %76 = add nuw i64 %60, 8
  %77 = add <4 x i64> %61, splat (i64 8)
  %78 = icmp eq i64 %76, 500000
  br i1 %78, label %79, label %59, !llvm.loop !21

79:                                               ; preds = %59
  %80 = add nuw nsw i32 %58, 1
  %81 = icmp eq i32 %80, 200
  br i1 %81, label %82, label %57, !llvm.loop !22

82:                                               ; preds = %79
  %83 = load i32, ptr %56, align 4, !tbaa !10
  %84 = xor <4 x i32> %75, %74
  %85 = tail call i32 @llvm.vector.reduce.xor.v4i32(<4 x i32> %84)
  %86 = xor i32 %83, %85
  tail call void @free(ptr noundef nonnull %3) #11
  %87 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %86)
  %88 = add nuw nsw i32 %2, 1
  %89 = icmp eq i32 %88, 5
  br i1 %89, label %90, label %1, !llvm.loop !23

90:                                               ; preds = %82
  ret i32 0
}

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.xor.v4i32(<4 x i32>) #9

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind memory(readwrite, argmem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { nounwind allocsize(0) }
attributes #11 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13, !15, !16}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !13, !16, !15}
!18 = distinct !{!18, !13, !15, !16}
!19 = distinct !{!19, !13, !16, !15}
!20 = distinct !{!20, !13, !15, !16}
!21 = distinct !{!21, !13, !15, !16}
!22 = distinct !{!22, !13}
!23 = distinct !{!23, !13}
