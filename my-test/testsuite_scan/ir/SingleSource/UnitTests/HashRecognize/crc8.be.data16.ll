; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc8.be.data16.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc8.be.data16.c"
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

16:                                               ; preds = %49
  ret i32 %137

17:                                               ; preds = %0, %49
  %18 = phi i64 [ 0, %0 ], [ %138, %49 ]
  %19 = phi i32 [ 0, %0 ], [ %137, %49 ]
  %20 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %18
  %21 = load i32, ptr %20, align 4, !tbaa !6
  %22 = trunc i32 %21 to i8
  %23 = sub nuw nsw i64 7, %18
  %24 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %23
  %25 = load i32, ptr %24, align 4, !tbaa !6
  %26 = load i8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 255), align 1, !tbaa !10
  %27 = icmp eq i8 %26, 0
  br i1 %27, label %28, label %49

28:                                               ; preds = %17
  store <16 x i8> %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 1), align 1, !tbaa !10
  store <16 x i8> %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 17), align 1, !tbaa !10
  store <16 x i8> %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 33), align 1, !tbaa !10
  store <8 x i8> %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 49), align 1, !tbaa !10
  store <4 x i8> %12, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 57), align 1, !tbaa !10
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 61), align 1, !tbaa !10
  store i8 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 62), align 2, !tbaa !10
  store i8 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 63), align 1, !tbaa !10
  %29 = load <16 x i8>, ptr @CRCTable, align 16, !tbaa !10
  %30 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  %31 = xor <16 x i8> %29, splat (i8 19)
  %32 = xor <16 x i8> %30, splat (i8 19)
  store <16 x i8> %31, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !10
  store <16 x i8> %32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !10
  %33 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !10
  %34 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  %35 = xor <16 x i8> %33, splat (i8 19)
  %36 = xor <16 x i8> %34, splat (i8 19)
  store <16 x i8> %35, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !10
  store <16 x i8> %36, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !10
  %37 = load <16 x i8>, ptr @CRCTable, align 16, !tbaa !10
  %38 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  %39 = xor <16 x i8> %37, splat (i8 38)
  %40 = xor <16 x i8> %38, splat (i8 38)
  store <16 x i8> %39, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 16, !tbaa !10
  store <16 x i8> %40, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !10
  %41 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !10
  %42 = load <16 x i8>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  %43 = xor <16 x i8> %41, splat (i8 38)
  %44 = xor <16 x i8> %42, splat (i8 38)
  store <16 x i8> %43, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 16, !tbaa !10
  store <16 x i8> %44, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !10
  %45 = xor <16 x i8> %31, splat (i8 38)
  %46 = xor <16 x i8> %32, splat (i8 38)
  store <16 x i8> %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 16, !tbaa !10
  store <16 x i8> %46, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !10
  %47 = xor <16 x i8> %35, splat (i8 38)
  %48 = xor <16 x i8> %36, splat (i8 38)
  store <16 x i8> %47, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 16, !tbaa !10
  store <16 x i8> %48, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !10
  br label %49

49:                                               ; preds = %28, %17
  %50 = trunc i32 %25 to i8
  %51 = xor i8 %50, %22
  %52 = zext i8 %51 to i64
  %53 = getelementptr inbounds nuw i8, ptr @CRCTable, i64 %52
  %54 = load i8, ptr %53, align 1, !tbaa !10
  %55 = zext i8 %54 to i64
  %56 = getelementptr inbounds nuw i8, ptr @CRCTable, i64 %55
  %57 = load i8, ptr %56, align 1, !tbaa !10
  %58 = shl i8 %22, 1
  %59 = xor i8 %58, 29
  %60 = icmp slt i8 %51, 0
  %61 = select i1 %60, i8 %59, i8 %58
  %62 = shl i8 %50, 1
  %63 = xor i8 %61, %62
  %64 = shl i8 %61, 1
  %65 = xor i8 %64, 29
  %66 = icmp slt i8 %63, 0
  %67 = select i1 %66, i8 %65, i8 %64
  %68 = shl i8 %50, 2
  %69 = xor i8 %67, %68
  %70 = shl i8 %67, 1
  %71 = xor i8 %70, 29
  %72 = icmp slt i8 %69, 0
  %73 = select i1 %72, i8 %71, i8 %70
  %74 = shl i8 %50, 3
  %75 = xor i8 %73, %74
  %76 = shl i8 %73, 1
  %77 = xor i8 %76, 29
  %78 = icmp slt i8 %75, 0
  %79 = select i1 %78, i8 %77, i8 %76
  %80 = shl i8 %50, 4
  %81 = xor i8 %79, %80
  %82 = shl i8 %79, 1
  %83 = xor i8 %82, 29
  %84 = icmp slt i8 %81, 0
  %85 = select i1 %84, i8 %83, i8 %82
  %86 = shl i8 %50, 5
  %87 = xor i8 %85, %86
  %88 = shl i8 %85, 1
  %89 = xor i8 %88, 29
  %90 = icmp slt i8 %87, 0
  %91 = select i1 %90, i8 %89, i8 %88
  %92 = shl i8 %50, 6
  %93 = xor i8 %91, %92
  %94 = shl i8 %91, 1
  %95 = xor i8 %94, 29
  %96 = icmp slt i8 %93, 0
  %97 = select i1 %96, i8 %95, i8 %94
  %98 = shl i8 %50, 7
  %99 = xor i8 %97, %98
  %100 = shl i8 %97, 1
  %101 = xor i8 %100, 29
  %102 = icmp slt i8 %99, 0
  %103 = select i1 %102, i8 %101, i8 %100
  %104 = shl i8 %103, 1
  %105 = xor i8 %104, 29
  %106 = icmp slt i8 %103, 0
  %107 = select i1 %106, i8 %105, i8 %104
  %108 = shl i8 %107, 1
  %109 = xor i8 %108, 29
  %110 = icmp slt i8 %107, 0
  %111 = select i1 %110, i8 %109, i8 %108
  %112 = shl i8 %111, 1
  %113 = xor i8 %112, 29
  %114 = icmp slt i8 %111, 0
  %115 = select i1 %114, i8 %113, i8 %112
  %116 = shl i8 %115, 1
  %117 = xor i8 %116, 29
  %118 = icmp slt i8 %115, 0
  %119 = select i1 %118, i8 %117, i8 %116
  %120 = shl i8 %119, 1
  %121 = xor i8 %120, 29
  %122 = icmp slt i8 %119, 0
  %123 = select i1 %122, i8 %121, i8 %120
  %124 = shl i8 %123, 1
  %125 = xor i8 %124, 29
  %126 = icmp slt i8 %123, 0
  %127 = select i1 %126, i8 %125, i8 %124
  %128 = shl i8 %127, 1
  %129 = xor i8 %128, 29
  %130 = icmp slt i8 %127, 0
  %131 = select i1 %130, i8 %129, i8 %128
  %132 = shl i8 %131, 1
  %133 = xor i8 %132, 29
  %134 = icmp slt i8 %131, 0
  %135 = select i1 %134, i8 %133, i8 %132
  %136 = icmp eq i8 %57, %135
  %137 = select i1 %136, i32 %19, i32 1
  %138 = add nuw nsw i64 %18, 1
  %139 = icmp eq i64 %138, 8
  br i1 %139, label %16, label %17, !llvm.loop !11
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
