; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/mode-dependent-address.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/mode-dependent-address.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@__const.main.correct = private unnamed_addr constant [96 x i32] [i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19, i32 16, i32 17, i32 18, i32 19, i32 24, i32 25, i32 26, i32 27, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35, i32 32, i32 33, i32 34, i32 35, i32 40, i32 41, i32 42, i32 43, i32 40, i32 41, i32 42, i32 43, i32 48, i32 49, i32 50, i32 51, i32 48, i32 49, i32 50, i32 51, i32 56, i32 57, i32 58, i32 59, i32 56, i32 57, i32 58, i32 59, i32 64, i32 65, i32 66, i32 67, i32 64, i32 65, i32 66, i32 67, i32 72, i32 73, i32 74, i32 75, i32 72, i32 73, i32 74, i32 75, i32 80, i32 81, i32 82, i32 83, i32 80, i32 81, i32 82, i32 83, i32 88, i32 89, i32 90, i32 91, i32 88, i32 89, i32 90, i32 91], align 64
@arg4 = dso_local local_unnamed_addr global [96 x i8] zeroinitializer, align 16
@arg1 = dso_local local_unnamed_addr global [96 x i16] zeroinitializer, align 32
@arg2 = dso_local local_unnamed_addr global [96 x i32] zeroinitializer, align 64
@arg3 = dso_local local_unnamed_addr global [96 x i64] zeroinitializer, align 128
@result = dso_local local_unnamed_addr global [96 x i8] zeroinitializer, align 16

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @f883b(ptr noundef writeonly captures(none) %0, ptr noalias noundef readonly captures(none) %1, ptr noalias noundef readonly captures(none) %2, ptr noalias noundef readonly captures(none) %3, ptr noalias noundef readnone captures(none) %4) local_unnamed_addr #0 {
  %6 = load <16 x i16>, ptr %1, align 2, !tbaa !6
  %7 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %6, <16 x i16> splat (i16 1))
  %8 = sext <16 x i16> %7 to <16 x i32>
  %9 = load <16 x i32>, ptr %2, align 4, !tbaa !10
  %10 = and <16 x i32> %9, splat (i32 31)
  %11 = ashr <16 x i32> %8, %10
  %12 = add nsw <16 x i32> %11, splat (i32 32)
  %13 = lshr <16 x i32> %12, splat (i32 7)
  %14 = load <16 x i64>, ptr %3, align 8, !tbaa !12
  %15 = trunc <16 x i64> %14 to <16 x i8>
  %16 = trunc <16 x i32> %13 to <16 x i8>
  %17 = or <16 x i8> %16, splat (i8 -5)
  %18 = and <16 x i8> %17, %15
  store <16 x i8> %18, ptr %0, align 1, !tbaa !14
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %20 = load <16 x i16>, ptr %19, align 2, !tbaa !6
  %21 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %20, <16 x i16> splat (i16 1))
  %22 = sext <16 x i16> %21 to <16 x i32>
  %23 = getelementptr inbounds nuw i8, ptr %2, i64 64
  %24 = load <16 x i32>, ptr %23, align 4, !tbaa !10
  %25 = and <16 x i32> %24, splat (i32 31)
  %26 = ashr <16 x i32> %22, %25
  %27 = add nsw <16 x i32> %26, splat (i32 32)
  %28 = lshr <16 x i32> %27, splat (i32 7)
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 128
  %30 = load <16 x i64>, ptr %29, align 8, !tbaa !12
  %31 = trunc <16 x i64> %30 to <16 x i8>
  %32 = trunc <16 x i32> %28 to <16 x i8>
  %33 = or <16 x i8> %32, splat (i8 -5)
  %34 = and <16 x i8> %33, %31
  %35 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store <16 x i8> %34, ptr %35, align 1, !tbaa !14
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %37 = load <16 x i16>, ptr %36, align 2, !tbaa !6
  %38 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %37, <16 x i16> splat (i16 1))
  %39 = sext <16 x i16> %38 to <16 x i32>
  %40 = getelementptr inbounds nuw i8, ptr %2, i64 128
  %41 = load <16 x i32>, ptr %40, align 4, !tbaa !10
  %42 = and <16 x i32> %41, splat (i32 31)
  %43 = ashr <16 x i32> %39, %42
  %44 = add nsw <16 x i32> %43, splat (i32 32)
  %45 = lshr <16 x i32> %44, splat (i32 7)
  %46 = getelementptr inbounds nuw i8, ptr %3, i64 256
  %47 = load <16 x i64>, ptr %46, align 8, !tbaa !12
  %48 = trunc <16 x i64> %47 to <16 x i8>
  %49 = trunc <16 x i32> %45 to <16 x i8>
  %50 = or <16 x i8> %49, splat (i8 -5)
  %51 = and <16 x i8> %50, %48
  %52 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store <16 x i8> %51, ptr %52, align 1, !tbaa !14
  %53 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %54 = load <16 x i16>, ptr %53, align 2, !tbaa !6
  %55 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %54, <16 x i16> splat (i16 1))
  %56 = sext <16 x i16> %55 to <16 x i32>
  %57 = getelementptr inbounds nuw i8, ptr %2, i64 192
  %58 = load <16 x i32>, ptr %57, align 4, !tbaa !10
  %59 = and <16 x i32> %58, splat (i32 31)
  %60 = ashr <16 x i32> %56, %59
  %61 = add nsw <16 x i32> %60, splat (i32 32)
  %62 = lshr <16 x i32> %61, splat (i32 7)
  %63 = getelementptr inbounds nuw i8, ptr %3, i64 384
  %64 = load <16 x i64>, ptr %63, align 8, !tbaa !12
  %65 = trunc <16 x i64> %64 to <16 x i8>
  %66 = trunc <16 x i32> %62 to <16 x i8>
  %67 = or <16 x i8> %66, splat (i8 -5)
  %68 = and <16 x i8> %67, %65
  %69 = getelementptr inbounds nuw i8, ptr %0, i64 48
  store <16 x i8> %68, ptr %69, align 1, !tbaa !14
  %70 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %71 = load <16 x i16>, ptr %70, align 2, !tbaa !6
  %72 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %71, <16 x i16> splat (i16 1))
  %73 = sext <16 x i16> %72 to <16 x i32>
  %74 = getelementptr inbounds nuw i8, ptr %2, i64 256
  %75 = load <16 x i32>, ptr %74, align 4, !tbaa !10
  %76 = and <16 x i32> %75, splat (i32 31)
  %77 = ashr <16 x i32> %73, %76
  %78 = add nsw <16 x i32> %77, splat (i32 32)
  %79 = lshr <16 x i32> %78, splat (i32 7)
  %80 = getelementptr inbounds nuw i8, ptr %3, i64 512
  %81 = load <16 x i64>, ptr %80, align 8, !tbaa !12
  %82 = trunc <16 x i64> %81 to <16 x i8>
  %83 = trunc <16 x i32> %79 to <16 x i8>
  %84 = or <16 x i8> %83, splat (i8 -5)
  %85 = and <16 x i8> %84, %82
  %86 = getelementptr inbounds nuw i8, ptr %0, i64 64
  store <16 x i8> %85, ptr %86, align 1, !tbaa !14
  %87 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %88 = load <16 x i16>, ptr %87, align 2, !tbaa !6
  %89 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %88, <16 x i16> splat (i16 1))
  %90 = sext <16 x i16> %89 to <16 x i32>
  %91 = getelementptr inbounds nuw i8, ptr %2, i64 320
  %92 = load <16 x i32>, ptr %91, align 4, !tbaa !10
  %93 = and <16 x i32> %92, splat (i32 31)
  %94 = ashr <16 x i32> %90, %93
  %95 = add nsw <16 x i32> %94, splat (i32 32)
  %96 = lshr <16 x i32> %95, splat (i32 7)
  %97 = getelementptr inbounds nuw i8, ptr %3, i64 640
  %98 = load <16 x i64>, ptr %97, align 8, !tbaa !12
  %99 = trunc <16 x i64> %98 to <16 x i8>
  %100 = trunc <16 x i32> %96 to <16 x i8>
  %101 = or <16 x i8> %100, splat (i8 -5)
  %102 = and <16 x i8> %101, %99
  %103 = getelementptr inbounds nuw i8, ptr %0, i64 80
  store <16 x i8> %102, ptr %103, align 1, !tbaa !14
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, ptr @arg4, align 16, !tbaa !14
  store <16 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, ptr @arg1, align 32, !tbaa !6
  store <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, ptr @arg2, align 64, !tbaa !10
  store <16 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>, ptr @arg3, align 128, !tbaa !12
  store <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, ptr getelementptr inbounds nuw (i8, ptr @arg4, i64 16), align 16, !tbaa !14
  store <16 x i16> <i16 16, i16 17, i16 18, i16 19, i16 20, i16 21, i16 22, i16 23, i16 24, i16 25, i16 26, i16 27, i16 28, i16 29, i16 30, i16 31>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 32), align 32, !tbaa !6
  store <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 64), align 64, !tbaa !10
  store <16 x i64> <i64 16, i64 17, i64 18, i64 19, i64 20, i64 21, i64 22, i64 23, i64 24, i64 25, i64 26, i64 27, i64 28, i64 29, i64 30, i64 31>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 128), align 128, !tbaa !12
  store <16 x i8> <i8 32, i8 33, i8 34, i8 35, i8 36, i8 37, i8 38, i8 39, i8 40, i8 41, i8 42, i8 43, i8 44, i8 45, i8 46, i8 47>, ptr getelementptr inbounds nuw (i8, ptr @arg4, i64 32), align 16, !tbaa !14
  store <16 x i16> <i16 32, i16 33, i16 34, i16 35, i16 36, i16 37, i16 38, i16 39, i16 40, i16 41, i16 42, i16 43, i16 44, i16 45, i16 46, i16 47>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 64), align 32, !tbaa !6
  store <16 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 128), align 64, !tbaa !10
  store <16 x i64> <i64 32, i64 33, i64 34, i64 35, i64 36, i64 37, i64 38, i64 39, i64 40, i64 41, i64 42, i64 43, i64 44, i64 45, i64 46, i64 47>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 256), align 128, !tbaa !12
  store <16 x i8> <i8 48, i8 49, i8 50, i8 51, i8 52, i8 53, i8 54, i8 55, i8 56, i8 57, i8 58, i8 59, i8 60, i8 61, i8 62, i8 63>, ptr getelementptr inbounds nuw (i8, ptr @arg4, i64 48), align 16, !tbaa !14
  store <16 x i16> <i16 48, i16 49, i16 50, i16 51, i16 52, i16 53, i16 54, i16 55, i16 56, i16 57, i16 58, i16 59, i16 60, i16 61, i16 62, i16 63>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 96), align 32, !tbaa !6
  store <16 x i32> <i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 192), align 64, !tbaa !10
  store <16 x i64> <i64 48, i64 49, i64 50, i64 51, i64 52, i64 53, i64 54, i64 55, i64 56, i64 57, i64 58, i64 59, i64 60, i64 61, i64 62, i64 63>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 384), align 128, !tbaa !12
  store <16 x i8> <i8 64, i8 65, i8 66, i8 67, i8 68, i8 69, i8 70, i8 71, i8 72, i8 73, i8 74, i8 75, i8 76, i8 77, i8 78, i8 79>, ptr getelementptr inbounds nuw (i8, ptr @arg4, i64 64), align 16, !tbaa !14
  store <16 x i16> <i16 64, i16 65, i16 66, i16 67, i16 68, i16 69, i16 70, i16 71, i16 72, i16 73, i16 74, i16 75, i16 76, i16 77, i16 78, i16 79>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 128), align 32, !tbaa !6
  store <16 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 256), align 64, !tbaa !10
  store <16 x i64> <i64 64, i64 65, i64 66, i64 67, i64 68, i64 69, i64 70, i64 71, i64 72, i64 73, i64 74, i64 75, i64 76, i64 77, i64 78, i64 79>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 512), align 128, !tbaa !12
  store <16 x i8> <i8 80, i8 81, i8 82, i8 83, i8 84, i8 85, i8 86, i8 87, i8 88, i8 89, i8 90, i8 91, i8 92, i8 93, i8 94, i8 95>, ptr getelementptr inbounds nuw (i8, ptr @arg4, i64 80), align 16, !tbaa !14
  store <16 x i16> <i16 80, i16 81, i16 82, i16 83, i16 84, i16 85, i16 86, i16 87, i16 88, i16 89, i16 90, i16 91, i16 92, i16 93, i16 94, i16 95>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 160), align 32, !tbaa !6
  store <16 x i32> <i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 320), align 64, !tbaa !10
  store <16 x i64> <i64 80, i64 81, i64 82, i64 83, i64 84, i64 85, i64 86, i64 87, i64 88, i64 89, i64 90, i64 91, i64 92, i64 93, i64 94, i64 95>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 640), align 128, !tbaa !12
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !18)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !20)
  %1 = load <16 x i16>, ptr @arg1, align 32, !tbaa !6, !alias.scope !15, !noalias !22
  %2 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %1, <16 x i16> splat (i16 1))
  %3 = sext <16 x i16> %2 to <16 x i32>
  %4 = load <16 x i32>, ptr @arg2, align 64, !tbaa !10, !alias.scope !18, !noalias !23
  %5 = and <16 x i32> %4, splat (i32 31)
  %6 = ashr <16 x i32> %3, %5
  %7 = add nsw <16 x i32> %6, splat (i32 32)
  %8 = lshr <16 x i32> %7, splat (i32 7)
  %9 = load <16 x i64>, ptr @arg3, align 128, !tbaa !12, !alias.scope !20, !noalias !24
  %10 = trunc <16 x i64> %9 to <16 x i8>
  %11 = trunc <16 x i32> %8 to <16 x i8>
  %12 = or <16 x i8> %11, splat (i8 -5)
  %13 = and <16 x i8> %12, %10
  store <16 x i8> %13, ptr @result, align 16, !tbaa !14, !noalias !25
  %14 = load <16 x i16>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 32), align 32, !tbaa !6, !alias.scope !15, !noalias !22
  %15 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %14, <16 x i16> splat (i16 1))
  %16 = sext <16 x i16> %15 to <16 x i32>
  %17 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 64), align 64, !tbaa !10, !alias.scope !18, !noalias !23
  %18 = and <16 x i32> %17, splat (i32 31)
  %19 = ashr <16 x i32> %16, %18
  %20 = add nsw <16 x i32> %19, splat (i32 32)
  %21 = lshr <16 x i32> %20, splat (i32 7)
  %22 = load <16 x i64>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 128), align 128, !tbaa !12, !alias.scope !20, !noalias !24
  %23 = trunc <16 x i64> %22 to <16 x i8>
  %24 = trunc <16 x i32> %21 to <16 x i8>
  %25 = or <16 x i8> %24, splat (i8 -5)
  %26 = and <16 x i8> %25, %23
  store <16 x i8> %26, ptr getelementptr inbounds nuw (i8, ptr @result, i64 16), align 16, !tbaa !14, !noalias !25
  %27 = load <16 x i16>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 64), align 32, !tbaa !6, !alias.scope !15, !noalias !22
  %28 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %27, <16 x i16> splat (i16 1))
  %29 = sext <16 x i16> %28 to <16 x i32>
  %30 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 128), align 64, !tbaa !10, !alias.scope !18, !noalias !23
  %31 = and <16 x i32> %30, splat (i32 31)
  %32 = ashr <16 x i32> %29, %31
  %33 = add nsw <16 x i32> %32, splat (i32 32)
  %34 = lshr <16 x i32> %33, splat (i32 7)
  %35 = load <16 x i64>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 256), align 128, !tbaa !12, !alias.scope !20, !noalias !24
  %36 = trunc <16 x i64> %35 to <16 x i8>
  %37 = trunc <16 x i32> %34 to <16 x i8>
  %38 = or <16 x i8> %37, splat (i8 -5)
  %39 = and <16 x i8> %38, %36
  store <16 x i8> %39, ptr getelementptr inbounds nuw (i8, ptr @result, i64 32), align 16, !tbaa !14, !noalias !25
  %40 = load <16 x i16>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 96), align 32, !tbaa !6, !alias.scope !15, !noalias !22
  %41 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %40, <16 x i16> splat (i16 1))
  %42 = sext <16 x i16> %41 to <16 x i32>
  %43 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 192), align 64, !tbaa !10, !alias.scope !18, !noalias !23
  %44 = and <16 x i32> %43, splat (i32 31)
  %45 = ashr <16 x i32> %42, %44
  %46 = add nsw <16 x i32> %45, splat (i32 32)
  %47 = lshr <16 x i32> %46, splat (i32 7)
  %48 = load <16 x i64>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 384), align 128, !tbaa !12, !alias.scope !20, !noalias !24
  %49 = trunc <16 x i64> %48 to <16 x i8>
  %50 = trunc <16 x i32> %47 to <16 x i8>
  %51 = or <16 x i8> %50, splat (i8 -5)
  %52 = and <16 x i8> %51, %49
  store <16 x i8> %52, ptr getelementptr inbounds nuw (i8, ptr @result, i64 48), align 16, !tbaa !14, !noalias !25
  %53 = load <16 x i16>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 128), align 32, !tbaa !6, !alias.scope !15, !noalias !22
  %54 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %53, <16 x i16> splat (i16 1))
  %55 = sext <16 x i16> %54 to <16 x i32>
  %56 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 256), align 64, !tbaa !10, !alias.scope !18, !noalias !23
  %57 = and <16 x i32> %56, splat (i32 31)
  %58 = ashr <16 x i32> %55, %57
  %59 = add nsw <16 x i32> %58, splat (i32 32)
  %60 = lshr <16 x i32> %59, splat (i32 7)
  %61 = load <16 x i64>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 512), align 128, !tbaa !12, !alias.scope !20, !noalias !24
  %62 = trunc <16 x i64> %61 to <16 x i8>
  %63 = trunc <16 x i32> %60 to <16 x i8>
  %64 = or <16 x i8> %63, splat (i8 -5)
  %65 = and <16 x i8> %64, %62
  store <16 x i8> %65, ptr getelementptr inbounds nuw (i8, ptr @result, i64 64), align 16, !tbaa !14, !noalias !25
  %66 = load <16 x i16>, ptr getelementptr inbounds nuw (i8, ptr @arg1, i64 160), align 32, !tbaa !6, !alias.scope !15, !noalias !22
  %67 = tail call <16 x i16> @llvm.smin.v16i16(<16 x i16> %66, <16 x i16> splat (i16 1))
  %68 = sext <16 x i16> %67 to <16 x i32>
  %69 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @arg2, i64 320), align 64, !tbaa !10, !alias.scope !18, !noalias !23
  %70 = and <16 x i32> %69, splat (i32 31)
  %71 = ashr <16 x i32> %68, %70
  %72 = add nsw <16 x i32> %71, splat (i32 32)
  %73 = lshr <16 x i32> %72, splat (i32 7)
  %74 = load <16 x i64>, ptr getelementptr inbounds nuw (i8, ptr @arg3, i64 640), align 128, !tbaa !12, !alias.scope !20, !noalias !24
  %75 = trunc <16 x i64> %74 to <16 x i8>
  %76 = trunc <16 x i32> %73 to <16 x i8>
  %77 = or <16 x i8> %76, splat (i8 -5)
  %78 = and <16 x i8> %77, %75
  store <16 x i8> %78, ptr getelementptr inbounds nuw (i8, ptr @result, i64 80), align 16, !tbaa !14, !noalias !25
  %79 = load <16 x i8>, ptr @result, align 16, !tbaa !14
  %80 = freeze <16 x i8> %79
  %81 = sext <16 x i8> %80 to <16 x i32>
  %82 = load <16 x i32>, ptr @__const.main.correct, align 64, !tbaa !10
  %83 = freeze <16 x i32> %82
  %84 = icmp ne <16 x i32> %83, %81
  %85 = bitcast <16 x i1> %84 to i16
  %86 = icmp eq i16 %85, 0
  br i1 %86, label %87, label %132, !llvm.loop !26

87:                                               ; preds = %0
  %88 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @result, i64 16), align 16, !tbaa !14
  %89 = freeze <16 x i8> %88
  %90 = sext <16 x i8> %89 to <16 x i32>
  %91 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @__const.main.correct, i64 64), align 64, !tbaa !10
  %92 = freeze <16 x i32> %91
  %93 = icmp ne <16 x i32> %92, %90
  %94 = bitcast <16 x i1> %93 to i16
  %95 = icmp eq i16 %94, 0
  br i1 %95, label %96, label %132, !llvm.loop !26

96:                                               ; preds = %87
  %97 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @result, i64 32), align 16, !tbaa !14
  %98 = freeze <16 x i8> %97
  %99 = sext <16 x i8> %98 to <16 x i32>
  %100 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @__const.main.correct, i64 128), align 64, !tbaa !10
  %101 = freeze <16 x i32> %100
  %102 = icmp ne <16 x i32> %101, %99
  %103 = bitcast <16 x i1> %102 to i16
  %104 = icmp eq i16 %103, 0
  br i1 %104, label %105, label %132, !llvm.loop !26

105:                                              ; preds = %96
  %106 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @result, i64 48), align 16, !tbaa !14
  %107 = freeze <16 x i8> %106
  %108 = sext <16 x i8> %107 to <16 x i32>
  %109 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @__const.main.correct, i64 192), align 64, !tbaa !10
  %110 = freeze <16 x i32> %109
  %111 = icmp ne <16 x i32> %110, %108
  %112 = bitcast <16 x i1> %111 to i16
  %113 = icmp eq i16 %112, 0
  br i1 %113, label %114, label %132, !llvm.loop !26

114:                                              ; preds = %105
  %115 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @result, i64 64), align 16, !tbaa !14
  %116 = freeze <16 x i8> %115
  %117 = sext <16 x i8> %116 to <16 x i32>
  %118 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @__const.main.correct, i64 256), align 64, !tbaa !10
  %119 = freeze <16 x i32> %118
  %120 = icmp ne <16 x i32> %119, %117
  %121 = bitcast <16 x i1> %120 to i16
  %122 = icmp eq i16 %121, 0
  br i1 %122, label %123, label %132, !llvm.loop !26

123:                                              ; preds = %114
  %124 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @result, i64 80), align 16, !tbaa !14
  %125 = freeze <16 x i8> %124
  %126 = sext <16 x i8> %125 to <16 x i32>
  %127 = load <16 x i32>, ptr getelementptr inbounds nuw (i8, ptr @__const.main.correct, i64 320), align 64, !tbaa !10
  %128 = freeze <16 x i32> %127
  %129 = icmp ne <16 x i32> %128, %126
  %130 = bitcast <16 x i1> %129 to i16
  %131 = icmp ne i16 %130, 0
  br i1 %131, label %132, label %133

132:                                              ; preds = %0, %87, %96, %105, %114, %123
  tail call void @abort() #5
  unreachable

133:                                              ; preds = %123
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x i16> @llvm.smin.v16i16(<16 x i16>, <16 x i16>) #4

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { cold noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"long", !8, i64 0}
!14 = !{!8, !8, i64 0}
!15 = !{!16}
!16 = distinct !{!16, !17, !"f883b: argument 0"}
!17 = distinct !{!17, !"f883b"}
!18 = !{!19}
!19 = distinct !{!19, !17, !"f883b: argument 1"}
!20 = !{!21}
!21 = distinct !{!21, !17, !"f883b: argument 2"}
!22 = !{!19, !21}
!23 = !{!16, !21}
!24 = !{!16, !19}
!25 = !{!16, !19, !21}
!26 = distinct !{!26, !27, !28, !29}
!27 = !{!"llvm.loop.mustprogress"}
!28 = !{!"llvm.loop.isvectorized", i32 1}
!29 = !{!"llvm.loop.unroll.runtime.disable"}
