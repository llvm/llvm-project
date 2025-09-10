; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc8.be.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc8.be.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.sample = internal unnamed_addr constant [8 x i32] [i32 0, i32 1, i32 11, i32 16, i32 129, i32 142, i32 196, i32 255], align 4
@CRCTable = internal unnamed_addr global [256 x i8] zeroinitializer, align 16

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  %1 = load i8, ptr @CRCTable, align 16
  %2 = insertelement <16 x i8> poison, i8 %1, i64 0
  %3 = shufflevector <16 x i8> %2, <16 x i8> poison, <16 x i32> zeroinitializer
  %4 = xor <16 x i8> %3, <i8 29, i8 58, i8 39, i8 116, i8 105, i8 78, i8 83, i8 -24, i8 -11, i8 -46, i8 -49, i8 -100, i8 -127, i8 -90, i8 -69, i8 -51>
  %5 = xor <16 x i8> %3, <i8 -48, i8 -9, i8 -22, i8 -71, i8 -92, i8 -125, i8 -98, i8 37, i8 56, i8 31, i8 2, i8 81, i8 76, i8 107, i8 118, i8 -121>
  %6 = xor <16 x i8> %3, <i8 -102, i8 -67, i8 -96, i8 -13, i8 -18, i8 -55, i8 -44, i8 111, i8 114, i8 85, i8 72, i8 27, i8 6, i8 33, i8 60, i8 74>
  %7 = insertelement <8 x i8> poison, i8 %1, i64 0
  %8 = shufflevector <8 x i8> %7, <8 x i8> poison, <8 x i32> zeroinitializer
  %9 = xor <8 x i8> %8, <i8 87, i8 112, i8 109, i8 62, i8 35, i8 4, i8 25, i8 -94>
  %10 = insertelement <4 x i8> poison, i8 %1, i64 0
  %11 = shufflevector <4 x i8> %10, <4 x i8> poison, <4 x i32> zeroinitializer
  %12 = xor <4 x i8> %11, <i8 -65, i8 -104, i8 -123, i8 -42>
  %13 = xor i8 %1, -53
  %14 = xor i8 %1, -20
  %15 = xor i8 %1, -15
  br label %17

16:                                               ; preds = %50
  ret i32 %102

17:                                               ; preds = %0, %50
  %18 = phi i64 [ 0, %0 ], [ %103, %50 ]
  %19 = phi i32 [ 0, %0 ], [ %102, %50 ]
  %20 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %18
  %21 = load i32, ptr %20, align 4, !tbaa !6
  %22 = trunc i32 %21 to i8
  %23 = sub nuw nsw i64 7, %18
  %24 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %23
  %25 = load i32, ptr %24, align 4, !tbaa !6
  %26 = trunc i32 %25 to i8
  %27 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 255), align 1, !tbaa !10
  %28 = icmp eq i8 %27, 0
  br i1 %28, label %29, label %50

29:                                               ; preds = %17
  store <16 x i8> %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 1), align 1, !tbaa !10
  store <16 x i8> %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 17), align 1, !tbaa !10
  store <16 x i8> %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 33), align 1, !tbaa !10
  store <8 x i8> %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 49), align 1, !tbaa !10
  store <4 x i8> %12, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 57), align 1, !tbaa !10
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 61), align 1, !tbaa !10
  store i8 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 62), align 2, !tbaa !10
  store i8 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 63), align 1, !tbaa !10
  %30 = load <16 x i8>, ptr @CRCTable, align 16, !tbaa !10
  %31 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  %32 = xor <16 x i8> %30, splat (i8 19)
  %33 = xor <16 x i8> %31, splat (i8 19)
  store <16 x i8> %32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !10
  store <16 x i8> %33, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !10
  %34 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !10
  %35 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  %36 = xor <16 x i8> %34, splat (i8 19)
  %37 = xor <16 x i8> %35, splat (i8 19)
  store <16 x i8> %36, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !10
  store <16 x i8> %37, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !10
  %38 = load <16 x i8>, ptr @CRCTable, align 16, !tbaa !10
  %39 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  %40 = xor <16 x i8> %38, splat (i8 38)
  %41 = xor <16 x i8> %39, splat (i8 38)
  store <16 x i8> %40, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 16, !tbaa !10
  store <16 x i8> %41, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !10
  %42 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !10
  %43 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  %44 = xor <16 x i8> %42, splat (i8 38)
  %45 = xor <16 x i8> %43, splat (i8 38)
  store <16 x i8> %44, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 16, !tbaa !10
  store <16 x i8> %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !10
  %46 = xor <16 x i8> %32, splat (i8 38)
  %47 = xor <16 x i8> %33, splat (i8 38)
  store <16 x i8> %46, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 16, !tbaa !10
  store <16 x i8> %47, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !10
  %48 = xor <16 x i8> %36, splat (i8 38)
  %49 = xor <16 x i8> %37, splat (i8 38)
  store <16 x i8> %48, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 16, !tbaa !10
  store <16 x i8> %49, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !10
  br label %50

50:                                               ; preds = %29, %17
  %51 = xor i8 %26, %22
  %52 = zext i8 %51 to i64
  %53 = getelementptr inbounds nuw i8, ptr @CRCTable, i64 %52
  %54 = load i8, ptr %53, align 1, !tbaa !10
  %55 = shl i8 %22, 1
  %56 = xor i8 %55, 29
  %57 = icmp slt i8 %51, 0
  %58 = select i1 %57, i8 %56, i8 %55
  %59 = shl i8 %26, 1
  %60 = xor i8 %58, %59
  %61 = shl i8 %58, 1
  %62 = xor i8 %61, 29
  %63 = icmp slt i8 %60, 0
  %64 = select i1 %63, i8 %62, i8 %61
  %65 = shl i8 %26, 2
  %66 = xor i8 %64, %65
  %67 = shl i8 %64, 1
  %68 = xor i8 %67, 29
  %69 = icmp slt i8 %66, 0
  %70 = select i1 %69, i8 %68, i8 %67
  %71 = shl i8 %26, 3
  %72 = xor i8 %70, %71
  %73 = shl i8 %70, 1
  %74 = xor i8 %73, 29
  %75 = icmp slt i8 %72, 0
  %76 = select i1 %75, i8 %74, i8 %73
  %77 = shl i8 %26, 4
  %78 = xor i8 %76, %77
  %79 = shl i8 %76, 1
  %80 = xor i8 %79, 29
  %81 = icmp slt i8 %78, 0
  %82 = select i1 %81, i8 %80, i8 %79
  %83 = shl i8 %26, 5
  %84 = xor i8 %82, %83
  %85 = shl i8 %82, 1
  %86 = xor i8 %85, 29
  %87 = icmp slt i8 %84, 0
  %88 = select i1 %87, i8 %86, i8 %85
  %89 = shl i8 %26, 6
  %90 = xor i8 %88, %89
  %91 = shl i8 %88, 1
  %92 = xor i8 %91, 29
  %93 = icmp slt i8 %90, 0
  %94 = select i1 %93, i8 %92, i8 %91
  %95 = shl i8 %26, 7
  %96 = xor i8 %94, %95
  %97 = shl i8 %94, 1
  %98 = xor i8 %97, 29
  %99 = icmp slt i8 %96, 0
  %100 = select i1 %99, i8 %98, i8 %97
  %101 = icmp eq i8 %54, %100
  %102 = select i1 %101, i32 %19, i32 1
  %103 = add nuw nsw i64 %18, 1
  %104 = icmp eq i64 %103, 8
  br i1 %104, label %16, label %17, !llvm.loop !11
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
