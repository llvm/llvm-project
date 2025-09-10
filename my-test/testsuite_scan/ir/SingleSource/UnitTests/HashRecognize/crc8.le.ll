; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc8.le.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc8.le.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.sample = internal unnamed_addr constant [8 x i32] [i32 0, i32 1, i32 11, i32 16, i32 129, i32 142, i32 196, i32 255], align 4
@CRCTable = internal unnamed_addr global [256 x i8] zeroinitializer, align 64

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  %1 = load i8, ptr @CRCTable, align 64
  %2 = xor i8 %1, 29
  %3 = xor i8 %1, 19
  %4 = xor i8 %1, 14
  %5 = xor i8 %1, 20
  %6 = xor i8 %1, 7
  %7 = xor i8 %1, 9
  %8 = xor i8 %1, 26
  %9 = xor i8 %1, 10
  %10 = xor i8 %1, 30
  %11 = xor i8 %1, 25
  %12 = xor i8 %1, 13
  %13 = xor i8 %1, 23
  %14 = xor i8 %1, 3
  %15 = xor i8 %1, 4
  %16 = xor i8 %1, 16
  %17 = xor i8 %1, 5
  %18 = xor i8 %1, 15
  %19 = xor i8 %1, 17
  %20 = xor i8 %1, 27
  %21 = xor i8 %1, 22
  %22 = xor i8 %1, 28
  %23 = xor i8 %1, 2
  %24 = xor i8 %1, 8
  %25 = xor i8 %1, 24
  %26 = xor i8 %1, 18
  %27 = xor i8 %1, 12
  %28 = xor i8 %1, 6
  %29 = xor i8 %1, 11
  %30 = xor i8 %1, 1
  %31 = xor i8 %1, 31
  %32 = xor i8 %1, 21
  br label %34

33:                                               ; preds = %301
  ret i32 %360

34:                                               ; preds = %0, %301
  %35 = phi i64 [ 0, %0 ], [ %361, %301 ]
  %36 = phi i32 [ 0, %0 ], [ %360, %301 ]
  %37 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %35
  %38 = load i32, ptr %37, align 4, !tbaa !6
  %39 = trunc i32 %38 to i8
  %40 = sub nuw nsw i64 7, %35
  %41 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %40
  %42 = load i32, ptr %41, align 4, !tbaa !6
  %43 = trunc i32 %42 to i8
  %44 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 255), align 1, !tbaa !10
  %45 = icmp eq i8 %44, 0
  br i1 %45, label %46, label %301

46:                                               ; preds = %34
  store i8 %2, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 64, !tbaa !10
  store i8 %3, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 64, !tbaa !10
  store i8 %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 64, !tbaa !10
  store i8 %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 32, !tbaa !10
  store i8 %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 32, !tbaa !10
  store i8 %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 32, !tbaa !10
  store i8 %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 32, !tbaa !10
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  store i8 %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  store i8 %11, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !10
  store i8 %12, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !10
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !10
  store i8 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !10
  store i8 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !10
  store i8 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !10
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 8), align 8, !tbaa !10
  store i8 %18, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 24), align 8, !tbaa !10
  store i8 %19, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 40), align 8, !tbaa !10
  store i8 %20, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 56), align 8, !tbaa !10
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 72), align 8, !tbaa !10
  store i8 %22, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 88), align 8, !tbaa !10
  store i8 %23, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 104), align 8, !tbaa !10
  store i8 %24, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 120), align 8, !tbaa !10
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 136), align 8, !tbaa !10
  store i8 %26, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 152), align 8, !tbaa !10
  store i8 %27, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 168), align 8, !tbaa !10
  store i8 %28, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 184), align 8, !tbaa !10
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 200), align 8, !tbaa !10
  store i8 %30, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 216), align 8, !tbaa !10
  store i8 %31, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 232), align 8, !tbaa !10
  store i8 %32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 248), align 8, !tbaa !10
  store i8 %31, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 4), align 4, !tbaa !10
  store i8 %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 12), align 4, !tbaa !10
  store i8 %32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 20), align 4, !tbaa !10
  store i8 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 28), align 4, !tbaa !10
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 36), align 4, !tbaa !10
  store i8 %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 44), align 4, !tbaa !10
  store i8 %30, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 52), align 4, !tbaa !10
  store i8 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 60), align 4, !tbaa !10
  store i8 %27, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 68), align 4, !tbaa !10
  store i8 %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 76), align 4, !tbaa !10
  store i8 %28, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 84), align 4, !tbaa !10
  store i8 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 92), align 4, !tbaa !10
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 100), align 4, !tbaa !10
  store i8 %2, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 108), align 4, !tbaa !10
  store i8 %26, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 116), align 4, !tbaa !10
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 124), align 4, !tbaa !10
  store i8 %23, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 132), align 4, !tbaa !10
  store i8 %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 140), align 4, !tbaa !10
  store i8 %24, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 148), align 4, !tbaa !10
  store i8 %12, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 156), align 4, !tbaa !10
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 164), align 4, !tbaa !10
  store i8 %3, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 172), align 4, !tbaa !10
  store i8 %22, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 180), align 4, !tbaa !10
  store i8 %11, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 188), align 4, !tbaa !10
  store i8 %19, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 196), align 4, !tbaa !10
  store i8 %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 204), align 4, !tbaa !10
  store i8 %20, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 212), align 4, !tbaa !10
  store i8 %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 220), align 4, !tbaa !10
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 228), align 4, !tbaa !10
  store i8 %1, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 236), align 4, !tbaa !10
  store i8 %18, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 244), align 4, !tbaa !10
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 252), align 4, !tbaa !10
  %47 = load <61 x i8>, ptr @CRCTable, align 64, !tbaa !10
  %48 = shufflevector <61 x i8> %47, <61 x i8> poison, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28, i32 32, i32 36, i32 40, i32 44, i32 48, i32 52, i32 56, i32 60>
  %49 = xor <16 x i8> %48, splat (i8 18)
  %50 = extractelement <16 x i8> %49, i64 0
  store i8 %50, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), align 2, !tbaa !10
  %51 = extractelement <16 x i8> %49, i64 1
  store i8 %51, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 6), align 2, !tbaa !10
  %52 = extractelement <16 x i8> %49, i64 2
  store i8 %52, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 10), align 2, !tbaa !10
  %53 = extractelement <16 x i8> %49, i64 3
  store i8 %53, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 14), align 2, !tbaa !10
  %54 = extractelement <16 x i8> %49, i64 4
  store i8 %54, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 18), align 2, !tbaa !10
  %55 = extractelement <16 x i8> %49, i64 5
  store i8 %55, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 22), align 2, !tbaa !10
  %56 = extractelement <16 x i8> %49, i64 6
  store i8 %56, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 26), align 2, !tbaa !10
  %57 = extractelement <16 x i8> %49, i64 7
  store i8 %57, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 30), align 2, !tbaa !10
  %58 = extractelement <16 x i8> %49, i64 8
  store i8 %58, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 34), align 2, !tbaa !10
  %59 = extractelement <16 x i8> %49, i64 9
  store i8 %59, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 38), align 2, !tbaa !10
  %60 = extractelement <16 x i8> %49, i64 10
  store i8 %60, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 42), align 2, !tbaa !10
  %61 = extractelement <16 x i8> %49, i64 11
  store i8 %61, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 46), align 2, !tbaa !10
  %62 = extractelement <16 x i8> %49, i64 12
  store i8 %62, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 50), align 2, !tbaa !10
  %63 = extractelement <16 x i8> %49, i64 13
  store i8 %63, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 54), align 2, !tbaa !10
  %64 = extractelement <16 x i8> %49, i64 14
  store i8 %64, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 58), align 2, !tbaa !10
  %65 = extractelement <16 x i8> %49, i64 15
  store i8 %65, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 62), align 2, !tbaa !10
  %66 = load <61 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 64, !tbaa !10
  %67 = shufflevector <61 x i8> %66, <61 x i8> poison, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28, i32 32, i32 36, i32 40, i32 44, i32 48, i32 52, i32 56, i32 60>
  %68 = xor <16 x i8> %67, splat (i8 18)
  %69 = extractelement <16 x i8> %68, i64 0
  store i8 %69, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 66), align 2, !tbaa !10
  %70 = extractelement <16 x i8> %68, i64 1
  store i8 %70, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 70), align 2, !tbaa !10
  %71 = extractelement <16 x i8> %68, i64 2
  store i8 %71, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 74), align 2, !tbaa !10
  %72 = extractelement <16 x i8> %68, i64 3
  store i8 %72, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 78), align 2, !tbaa !10
  %73 = extractelement <16 x i8> %68, i64 4
  store i8 %73, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 82), align 2, !tbaa !10
  %74 = extractelement <16 x i8> %68, i64 5
  store i8 %74, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 86), align 2, !tbaa !10
  %75 = extractelement <16 x i8> %68, i64 6
  store i8 %75, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 90), align 2, !tbaa !10
  %76 = extractelement <16 x i8> %68, i64 7
  store i8 %76, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 94), align 2, !tbaa !10
  %77 = extractelement <16 x i8> %68, i64 8
  store i8 %77, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 98), align 2, !tbaa !10
  %78 = extractelement <16 x i8> %68, i64 9
  store i8 %78, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 102), align 2, !tbaa !10
  %79 = extractelement <16 x i8> %68, i64 10
  store i8 %79, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 106), align 2, !tbaa !10
  %80 = extractelement <16 x i8> %68, i64 11
  store i8 %80, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 110), align 2, !tbaa !10
  %81 = extractelement <16 x i8> %68, i64 12
  store i8 %81, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 114), align 2, !tbaa !10
  %82 = extractelement <16 x i8> %68, i64 13
  store i8 %82, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 118), align 2, !tbaa !10
  %83 = extractelement <16 x i8> %68, i64 14
  store i8 %83, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 122), align 2, !tbaa !10
  %84 = extractelement <16 x i8> %68, i64 15
  store i8 %84, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 126), align 2, !tbaa !10
  %85 = load <61 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 64, !tbaa !10
  %86 = shufflevector <61 x i8> %85, <61 x i8> poison, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28, i32 32, i32 36, i32 40, i32 44, i32 48, i32 52, i32 56, i32 60>
  %87 = xor <16 x i8> %86, splat (i8 18)
  %88 = extractelement <16 x i8> %87, i64 0
  store i8 %88, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 130), align 2, !tbaa !10
  %89 = extractelement <16 x i8> %87, i64 1
  store i8 %89, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 134), align 2, !tbaa !10
  %90 = extractelement <16 x i8> %87, i64 2
  store i8 %90, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 138), align 2, !tbaa !10
  %91 = extractelement <16 x i8> %87, i64 3
  store i8 %91, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 142), align 2, !tbaa !10
  %92 = extractelement <16 x i8> %87, i64 4
  store i8 %92, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 146), align 2, !tbaa !10
  %93 = extractelement <16 x i8> %87, i64 5
  store i8 %93, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 150), align 2, !tbaa !10
  %94 = extractelement <16 x i8> %87, i64 6
  store i8 %94, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 154), align 2, !tbaa !10
  %95 = extractelement <16 x i8> %87, i64 7
  store i8 %95, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 158), align 2, !tbaa !10
  %96 = extractelement <16 x i8> %87, i64 8
  store i8 %96, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 162), align 2, !tbaa !10
  %97 = extractelement <16 x i8> %87, i64 9
  store i8 %97, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 166), align 2, !tbaa !10
  %98 = extractelement <16 x i8> %87, i64 10
  store i8 %98, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 170), align 2, !tbaa !10
  %99 = extractelement <16 x i8> %87, i64 11
  store i8 %99, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 174), align 2, !tbaa !10
  %100 = extractelement <16 x i8> %87, i64 12
  store i8 %100, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 178), align 2, !tbaa !10
  %101 = extractelement <16 x i8> %87, i64 13
  store i8 %101, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 182), align 2, !tbaa !10
  %102 = extractelement <16 x i8> %87, i64 14
  store i8 %102, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 186), align 2, !tbaa !10
  %103 = extractelement <16 x i8> %87, i64 15
  store i8 %103, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 190), align 2, !tbaa !10
  %104 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 64, !tbaa !10
  %105 = xor i8 %104, 18
  store i8 %105, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 194), align 2, !tbaa !10
  %106 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 196), align 4, !tbaa !10
  %107 = xor i8 %106, 18
  store i8 %107, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 198), align 2, !tbaa !10
  %108 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 200), align 8, !tbaa !10
  %109 = xor i8 %108, 18
  store i8 %109, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 202), align 2, !tbaa !10
  %110 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 204), align 4, !tbaa !10
  %111 = xor i8 %110, 18
  store i8 %111, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 206), align 2, !tbaa !10
  %112 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !10
  %113 = xor i8 %112, 18
  store i8 %113, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 210), align 2, !tbaa !10
  %114 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 212), align 4, !tbaa !10
  %115 = xor i8 %114, 18
  store i8 %115, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 214), align 2, !tbaa !10
  %116 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 216), align 8, !tbaa !10
  %117 = xor i8 %116, 18
  store i8 %117, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 218), align 2, !tbaa !10
  %118 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 220), align 4, !tbaa !10
  %119 = xor i8 %118, 18
  store i8 %119, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 222), align 2, !tbaa !10
  %120 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 32, !tbaa !10
  %121 = xor i8 %120, 18
  store i8 %121, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 226), align 2, !tbaa !10
  %122 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 228), align 4, !tbaa !10
  %123 = xor i8 %122, 18
  store i8 %123, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 230), align 2, !tbaa !10
  %124 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 232), align 8, !tbaa !10
  %125 = xor i8 %124, 18
  store i8 %125, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 234), align 2, !tbaa !10
  %126 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 236), align 4, !tbaa !10
  %127 = xor i8 %126, 18
  store i8 %127, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 238), align 2, !tbaa !10
  %128 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !10
  %129 = xor i8 %128, 18
  store i8 %129, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 242), align 2, !tbaa !10
  %130 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 244), align 4, !tbaa !10
  %131 = xor i8 %130, 18
  store i8 %131, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 246), align 2, !tbaa !10
  %132 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 248), align 8, !tbaa !10
  %133 = xor i8 %132, 18
  store i8 %133, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 250), align 2, !tbaa !10
  %134 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 252), align 4, !tbaa !10
  %135 = xor i8 %134, 18
  store i8 %135, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 254), align 2, !tbaa !10
  %136 = load <31 x i8>, ptr @CRCTable, align 64, !tbaa !10
  %137 = shufflevector <31 x i8> %136, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %138 = xor <16 x i8> %137, splat (i8 9)
  %139 = extractelement <16 x i8> %138, i64 0
  store i8 %139, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 1), align 1, !tbaa !10
  %140 = extractelement <16 x i8> %138, i64 1
  store i8 %140, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 3), align 1, !tbaa !10
  %141 = extractelement <16 x i8> %138, i64 2
  store i8 %141, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 5), align 1, !tbaa !10
  %142 = extractelement <16 x i8> %138, i64 3
  store i8 %142, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 7), align 1, !tbaa !10
  %143 = extractelement <16 x i8> %138, i64 4
  store i8 %143, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 9), align 1, !tbaa !10
  %144 = extractelement <16 x i8> %138, i64 5
  store i8 %144, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 11), align 1, !tbaa !10
  %145 = extractelement <16 x i8> %138, i64 6
  store i8 %145, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 13), align 1, !tbaa !10
  %146 = extractelement <16 x i8> %138, i64 7
  store i8 %146, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 15), align 1, !tbaa !10
  %147 = extractelement <16 x i8> %138, i64 8
  store i8 %147, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 17), align 1, !tbaa !10
  %148 = extractelement <16 x i8> %138, i64 9
  store i8 %148, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 19), align 1, !tbaa !10
  %149 = extractelement <16 x i8> %138, i64 10
  store i8 %149, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 21), align 1, !tbaa !10
  %150 = extractelement <16 x i8> %138, i64 11
  store i8 %150, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 23), align 1, !tbaa !10
  %151 = extractelement <16 x i8> %138, i64 12
  store i8 %151, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 25), align 1, !tbaa !10
  %152 = extractelement <16 x i8> %138, i64 13
  store i8 %152, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 27), align 1, !tbaa !10
  %153 = extractelement <16 x i8> %138, i64 14
  store i8 %153, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 29), align 1, !tbaa !10
  %154 = extractelement <16 x i8> %138, i64 15
  store i8 %154, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 31), align 1, !tbaa !10
  %155 = load <31 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 32, !tbaa !10
  %156 = shufflevector <31 x i8> %155, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %157 = xor <16 x i8> %156, splat (i8 9)
  %158 = extractelement <16 x i8> %157, i64 0
  store i8 %158, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 33), align 1, !tbaa !10
  %159 = extractelement <16 x i8> %157, i64 1
  store i8 %159, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 35), align 1, !tbaa !10
  %160 = extractelement <16 x i8> %157, i64 2
  store i8 %160, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 37), align 1, !tbaa !10
  %161 = extractelement <16 x i8> %157, i64 3
  store i8 %161, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 39), align 1, !tbaa !10
  %162 = extractelement <16 x i8> %157, i64 4
  store i8 %162, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 41), align 1, !tbaa !10
  %163 = extractelement <16 x i8> %157, i64 5
  store i8 %163, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 43), align 1, !tbaa !10
  %164 = extractelement <16 x i8> %157, i64 6
  store i8 %164, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 45), align 1, !tbaa !10
  %165 = extractelement <16 x i8> %157, i64 7
  store i8 %165, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 47), align 1, !tbaa !10
  %166 = extractelement <16 x i8> %157, i64 8
  store i8 %166, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 49), align 1, !tbaa !10
  %167 = extractelement <16 x i8> %157, i64 9
  store i8 %167, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 51), align 1, !tbaa !10
  %168 = extractelement <16 x i8> %157, i64 10
  store i8 %168, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 53), align 1, !tbaa !10
  %169 = extractelement <16 x i8> %157, i64 11
  store i8 %169, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 55), align 1, !tbaa !10
  %170 = extractelement <16 x i8> %157, i64 12
  store i8 %170, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 57), align 1, !tbaa !10
  %171 = extractelement <16 x i8> %157, i64 13
  store i8 %171, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 59), align 1, !tbaa !10
  %172 = extractelement <16 x i8> %157, i64 14
  store i8 %172, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 61), align 1, !tbaa !10
  %173 = extractelement <16 x i8> %157, i64 15
  store i8 %173, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 63), align 1, !tbaa !10
  %174 = load <31 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 64, !tbaa !10
  %175 = shufflevector <31 x i8> %174, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %176 = xor <16 x i8> %175, splat (i8 9)
  %177 = extractelement <16 x i8> %176, i64 0
  store i8 %177, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 65), align 1, !tbaa !10
  %178 = extractelement <16 x i8> %176, i64 1
  store i8 %178, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 67), align 1, !tbaa !10
  %179 = extractelement <16 x i8> %176, i64 2
  store i8 %179, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 69), align 1, !tbaa !10
  %180 = extractelement <16 x i8> %176, i64 3
  store i8 %180, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 71), align 1, !tbaa !10
  %181 = extractelement <16 x i8> %176, i64 4
  store i8 %181, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 73), align 1, !tbaa !10
  %182 = extractelement <16 x i8> %176, i64 5
  store i8 %182, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 75), align 1, !tbaa !10
  %183 = extractelement <16 x i8> %176, i64 6
  store i8 %183, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 77), align 1, !tbaa !10
  %184 = extractelement <16 x i8> %176, i64 7
  store i8 %184, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 79), align 1, !tbaa !10
  %185 = extractelement <16 x i8> %176, i64 8
  store i8 %185, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 81), align 1, !tbaa !10
  %186 = extractelement <16 x i8> %176, i64 9
  store i8 %186, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 83), align 1, !tbaa !10
  %187 = extractelement <16 x i8> %176, i64 10
  store i8 %187, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 85), align 1, !tbaa !10
  %188 = extractelement <16 x i8> %176, i64 11
  store i8 %188, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 87), align 1, !tbaa !10
  %189 = extractelement <16 x i8> %176, i64 12
  store i8 %189, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 89), align 1, !tbaa !10
  %190 = extractelement <16 x i8> %176, i64 13
  store i8 %190, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 91), align 1, !tbaa !10
  %191 = extractelement <16 x i8> %176, i64 14
  store i8 %191, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 93), align 1, !tbaa !10
  %192 = extractelement <16 x i8> %176, i64 15
  store i8 %192, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 95), align 1, !tbaa !10
  %193 = load <31 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 32, !tbaa !10
  %194 = shufflevector <31 x i8> %193, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %195 = xor <16 x i8> %194, splat (i8 9)
  %196 = extractelement <16 x i8> %195, i64 0
  store i8 %196, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 97), align 1, !tbaa !10
  %197 = extractelement <16 x i8> %195, i64 1
  store i8 %197, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 99), align 1, !tbaa !10
  %198 = extractelement <16 x i8> %195, i64 2
  store i8 %198, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 101), align 1, !tbaa !10
  %199 = extractelement <16 x i8> %195, i64 3
  store i8 %199, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 103), align 1, !tbaa !10
  %200 = extractelement <16 x i8> %195, i64 4
  store i8 %200, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 105), align 1, !tbaa !10
  %201 = extractelement <16 x i8> %195, i64 5
  store i8 %201, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 107), align 1, !tbaa !10
  %202 = extractelement <16 x i8> %195, i64 6
  store i8 %202, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 109), align 1, !tbaa !10
  %203 = extractelement <16 x i8> %195, i64 7
  store i8 %203, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 111), align 1, !tbaa !10
  %204 = extractelement <16 x i8> %195, i64 8
  store i8 %204, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 113), align 1, !tbaa !10
  %205 = extractelement <16 x i8> %195, i64 9
  store i8 %205, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 115), align 1, !tbaa !10
  %206 = extractelement <16 x i8> %195, i64 10
  store i8 %206, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 117), align 1, !tbaa !10
  %207 = extractelement <16 x i8> %195, i64 11
  store i8 %207, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 119), align 1, !tbaa !10
  %208 = extractelement <16 x i8> %195, i64 12
  store i8 %208, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 121), align 1, !tbaa !10
  %209 = extractelement <16 x i8> %195, i64 13
  store i8 %209, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 123), align 1, !tbaa !10
  %210 = extractelement <16 x i8> %195, i64 14
  store i8 %210, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 125), align 1, !tbaa !10
  %211 = extractelement <16 x i8> %195, i64 15
  store i8 %211, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 127), align 1, !tbaa !10
  %212 = load <31 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 64, !tbaa !10
  %213 = shufflevector <31 x i8> %212, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %214 = xor <16 x i8> %213, splat (i8 9)
  %215 = extractelement <16 x i8> %214, i64 0
  store i8 %215, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 129), align 1, !tbaa !10
  %216 = extractelement <16 x i8> %214, i64 1
  store i8 %216, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 131), align 1, !tbaa !10
  %217 = extractelement <16 x i8> %214, i64 2
  store i8 %217, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 133), align 1, !tbaa !10
  %218 = extractelement <16 x i8> %214, i64 3
  store i8 %218, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 135), align 1, !tbaa !10
  %219 = extractelement <16 x i8> %214, i64 4
  store i8 %219, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 137), align 1, !tbaa !10
  %220 = extractelement <16 x i8> %214, i64 5
  store i8 %220, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 139), align 1, !tbaa !10
  %221 = extractelement <16 x i8> %214, i64 6
  store i8 %221, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 141), align 1, !tbaa !10
  %222 = extractelement <16 x i8> %214, i64 7
  store i8 %222, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 143), align 1, !tbaa !10
  %223 = extractelement <16 x i8> %214, i64 8
  store i8 %223, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 145), align 1, !tbaa !10
  %224 = extractelement <16 x i8> %214, i64 9
  store i8 %224, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 147), align 1, !tbaa !10
  %225 = extractelement <16 x i8> %214, i64 10
  store i8 %225, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 149), align 1, !tbaa !10
  %226 = extractelement <16 x i8> %214, i64 11
  store i8 %226, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 151), align 1, !tbaa !10
  %227 = extractelement <16 x i8> %214, i64 12
  store i8 %227, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 153), align 1, !tbaa !10
  %228 = extractelement <16 x i8> %214, i64 13
  store i8 %228, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 155), align 1, !tbaa !10
  %229 = extractelement <16 x i8> %214, i64 14
  store i8 %229, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 157), align 1, !tbaa !10
  %230 = extractelement <16 x i8> %214, i64 15
  store i8 %230, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 159), align 1, !tbaa !10
  %231 = load <31 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 32, !tbaa !10
  %232 = shufflevector <31 x i8> %231, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %233 = xor <16 x i8> %232, splat (i8 9)
  %234 = extractelement <16 x i8> %233, i64 0
  store i8 %234, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 161), align 1, !tbaa !10
  %235 = extractelement <16 x i8> %233, i64 1
  store i8 %235, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 163), align 1, !tbaa !10
  %236 = extractelement <16 x i8> %233, i64 2
  store i8 %236, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 165), align 1, !tbaa !10
  %237 = extractelement <16 x i8> %233, i64 3
  store i8 %237, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 167), align 1, !tbaa !10
  %238 = extractelement <16 x i8> %233, i64 4
  store i8 %238, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 169), align 1, !tbaa !10
  %239 = extractelement <16 x i8> %233, i64 5
  store i8 %239, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 171), align 1, !tbaa !10
  %240 = extractelement <16 x i8> %233, i64 6
  store i8 %240, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 173), align 1, !tbaa !10
  %241 = extractelement <16 x i8> %233, i64 7
  store i8 %241, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 175), align 1, !tbaa !10
  %242 = extractelement <16 x i8> %233, i64 8
  store i8 %242, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 177), align 1, !tbaa !10
  %243 = extractelement <16 x i8> %233, i64 9
  store i8 %243, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 179), align 1, !tbaa !10
  %244 = extractelement <16 x i8> %233, i64 10
  store i8 %244, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 181), align 1, !tbaa !10
  %245 = extractelement <16 x i8> %233, i64 11
  store i8 %245, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 183), align 1, !tbaa !10
  %246 = extractelement <16 x i8> %233, i64 12
  store i8 %246, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 185), align 1, !tbaa !10
  %247 = extractelement <16 x i8> %233, i64 13
  store i8 %247, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 187), align 1, !tbaa !10
  %248 = extractelement <16 x i8> %233, i64 14
  store i8 %248, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 189), align 1, !tbaa !10
  %249 = extractelement <16 x i8> %233, i64 15
  store i8 %249, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 191), align 1, !tbaa !10
  %250 = load <31 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 64, !tbaa !10
  %251 = shufflevector <31 x i8> %250, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %252 = xor <16 x i8> %251, splat (i8 9)
  %253 = extractelement <16 x i8> %252, i64 0
  store i8 %253, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 193), align 1, !tbaa !10
  %254 = extractelement <16 x i8> %252, i64 1
  store i8 %254, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 195), align 1, !tbaa !10
  %255 = extractelement <16 x i8> %252, i64 2
  store i8 %255, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 197), align 1, !tbaa !10
  %256 = extractelement <16 x i8> %252, i64 3
  store i8 %256, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 199), align 1, !tbaa !10
  %257 = extractelement <16 x i8> %252, i64 4
  store i8 %257, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 201), align 1, !tbaa !10
  %258 = extractelement <16 x i8> %252, i64 5
  store i8 %258, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 203), align 1, !tbaa !10
  %259 = extractelement <16 x i8> %252, i64 6
  store i8 %259, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 205), align 1, !tbaa !10
  %260 = extractelement <16 x i8> %252, i64 7
  store i8 %260, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 207), align 1, !tbaa !10
  %261 = extractelement <16 x i8> %252, i64 8
  store i8 %261, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 209), align 1, !tbaa !10
  %262 = extractelement <16 x i8> %252, i64 9
  store i8 %262, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 211), align 1, !tbaa !10
  %263 = extractelement <16 x i8> %252, i64 10
  store i8 %263, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 213), align 1, !tbaa !10
  %264 = extractelement <16 x i8> %252, i64 11
  store i8 %264, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 215), align 1, !tbaa !10
  %265 = extractelement <16 x i8> %252, i64 12
  store i8 %265, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 217), align 1, !tbaa !10
  %266 = extractelement <16 x i8> %252, i64 13
  store i8 %266, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 219), align 1, !tbaa !10
  %267 = extractelement <16 x i8> %252, i64 14
  store i8 %267, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 221), align 1, !tbaa !10
  %268 = extractelement <16 x i8> %252, i64 15
  store i8 %268, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 223), align 1, !tbaa !10
  %269 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 32, !tbaa !10
  %270 = xor i8 %269, 9
  store i8 %270, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 225), align 1, !tbaa !10
  %271 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 226), align 2, !tbaa !10
  %272 = xor i8 %271, 9
  store i8 %272, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 227), align 1, !tbaa !10
  %273 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 228), align 4, !tbaa !10
  %274 = xor i8 %273, 9
  store i8 %274, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 229), align 1, !tbaa !10
  %275 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 230), align 2, !tbaa !10
  %276 = xor i8 %275, 9
  store i8 %276, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 231), align 1, !tbaa !10
  %277 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 232), align 8, !tbaa !10
  %278 = xor i8 %277, 9
  store i8 %278, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 233), align 1, !tbaa !10
  %279 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 234), align 2, !tbaa !10
  %280 = xor i8 %279, 9
  store i8 %280, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 235), align 1, !tbaa !10
  %281 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 236), align 4, !tbaa !10
  %282 = xor i8 %281, 9
  store i8 %282, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 237), align 1, !tbaa !10
  %283 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 238), align 2, !tbaa !10
  %284 = xor i8 %283, 9
  store i8 %284, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 239), align 1, !tbaa !10
  %285 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !10
  %286 = xor i8 %285, 9
  store i8 %286, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 241), align 1, !tbaa !10
  %287 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 242), align 2, !tbaa !10
  %288 = xor i8 %287, 9
  store i8 %288, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 243), align 1, !tbaa !10
  %289 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 244), align 4, !tbaa !10
  %290 = xor i8 %289, 9
  store i8 %290, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 245), align 1, !tbaa !10
  %291 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 246), align 2, !tbaa !10
  %292 = xor i8 %291, 9
  store i8 %292, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 247), align 1, !tbaa !10
  %293 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 248), align 8, !tbaa !10
  %294 = xor i8 %293, 9
  store i8 %294, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 249), align 1, !tbaa !10
  %295 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 250), align 2, !tbaa !10
  %296 = xor i8 %295, 9
  store i8 %296, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 251), align 1, !tbaa !10
  %297 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 252), align 4, !tbaa !10
  %298 = xor i8 %297, 9
  store i8 %298, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 253), align 1, !tbaa !10
  %299 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 254), align 2, !tbaa !10
  %300 = xor i8 %299, 9
  store i8 %300, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 255), align 1, !tbaa !10
  br label %301

301:                                              ; preds = %46, %34
  %302 = xor i8 %43, %39
  %303 = zext i8 %302 to i64
  %304 = getelementptr inbounds nuw i8, ptr @CRCTable, i64 %303
  %305 = load i8, ptr %304, align 1, !tbaa !10
  %306 = lshr i8 %39, 1
  %307 = and i8 %302, 1
  %308 = icmp eq i8 %307, 0
  %309 = xor i8 %306, 29
  %310 = select i1 %308, i8 %306, i8 %309
  %311 = lshr i8 %43, 1
  %312 = xor i8 %310, %311
  %313 = lshr i8 %310, 1
  %314 = and i8 %312, 1
  %315 = icmp eq i8 %314, 0
  %316 = xor i8 %313, 29
  %317 = select i1 %315, i8 %313, i8 %316
  %318 = lshr i8 %43, 2
  %319 = xor i8 %317, %318
  %320 = lshr i8 %317, 1
  %321 = and i8 %319, 1
  %322 = icmp eq i8 %321, 0
  %323 = xor i8 %320, 29
  %324 = select i1 %322, i8 %320, i8 %323
  %325 = lshr i8 %43, 3
  %326 = xor i8 %324, %325
  %327 = lshr i8 %324, 1
  %328 = and i8 %326, 1
  %329 = icmp eq i8 %328, 0
  %330 = xor i8 %327, 29
  %331 = select i1 %329, i8 %327, i8 %330
  %332 = lshr i8 %43, 4
  %333 = xor i8 %331, %332
  %334 = lshr i8 %331, 1
  %335 = and i8 %333, 1
  %336 = icmp eq i8 %335, 0
  %337 = xor i8 %334, 29
  %338 = select i1 %336, i8 %334, i8 %337
  %339 = lshr i8 %43, 5
  %340 = xor i8 %338, %339
  %341 = lshr i8 %338, 1
  %342 = and i8 %340, 1
  %343 = icmp eq i8 %342, 0
  %344 = xor i8 %341, 29
  %345 = select i1 %343, i8 %341, i8 %344
  %346 = lshr i8 %43, 6
  %347 = xor i8 %345, %346
  %348 = lshr i8 %345, 1
  %349 = and i8 %347, 1
  %350 = icmp eq i8 %349, 0
  %351 = xor i8 %348, 29
  %352 = select i1 %350, i8 %348, i8 %351
  %353 = lshr i8 %43, 7
  %354 = lshr i8 %352, 1
  %355 = and i8 %352, 1
  %356 = icmp eq i8 %353, %355
  %357 = xor i8 %354, 29
  %358 = select i1 %356, i8 %354, i8 %357
  %359 = icmp eq i8 %305, %358
  %360 = select i1 %359, i32 %36, i32 1
  %361 = add nuw nsw i64 %35, 1
  %362 = icmp eq i64 %361, 8
  br i1 %362, label %33, label %34, !llvm.loop !11
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
