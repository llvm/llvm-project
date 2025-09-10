; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.be.data8.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.be.data8.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.sample = internal unnamed_addr constant [8 x i32] [i32 0, i32 1, i32 11, i32 16, i32 129, i32 142, i32 196, i32 255], align 4
@CRCTable = internal unnamed_addr global [256 x i16] zeroinitializer, align 16

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  %1 = load i16, ptr @CRCTable, align 16
  %2 = insertelement <8 x i16> poison, i16 %1, i64 0
  %3 = shufflevector <8 x i16> %2, <8 x i16> poison, <8 x i32> zeroinitializer
  %4 = xor <8 x i16> %3, <i16 4129, i16 8258, i16 12387, i16 16516, i16 20645, i16 24774, i16 28903, i16 -32504>
  %5 = xor <8 x i16> %3, <i16 -28375, i16 -24246, i16 -20117, i16 -15988, i16 -11859, i16 -7730, i16 -3601, i16 4657>
  %6 = xor <8 x i16> %3, <i16 528, i16 12915, i16 8786, i16 21173, i16 17044, i16 29431, i16 25302, i16 -27847>
  %7 = xor <8 x i16> %3, <i16 -31976, i16 -19589, i16 -23718, i16 -11331, i16 -15460, i16 -3073, i16 -7202, i16 9314>
  %8 = xor <8 x i16> %3, <i16 13379, i16 1056, i16 5121, i16 25830, i16 29895, i16 17572, i16 21637, i16 -23190>
  %9 = xor <8 x i16> %3, <i16 -19125, i16 -31448, i16 -27383, i16 -6674, i16 -2609, i16 -14932, i16 -10867, i16 13907>
  %10 = xor <8 x i16> %3, <i16 9842, i16 5649, i16 1584, i16 30423, i16 26358, i16 22165, i16 18100, i16 -18597>
  %11 = insertelement <4 x i16> poison, i16 %1, i64 0
  %12 = shufflevector <4 x i16> %11, <4 x i16> poison, <4 x i32> zeroinitializer
  %13 = xor <4 x i16> %12, <i16 -22662, i16 -26855, i16 -30920, i16 -2081>
  %14 = xor i16 %1, -6146
  %15 = xor i16 %1, -10339
  %16 = xor i16 %1, -14404
  %17 = load i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !6
  %18 = icmp eq i16 %17, 0
  br i1 %18, label %26, label %19

19:                                               ; preds = %0
  %20 = icmp ne i16 %1, 0
  %21 = zext i1 %20 to i32
  br label %22

22:                                               ; preds = %75, %19
  %23 = phi i32 [ %21, %19 ], [ %115, %75 ]
  ret i32 %23

24:                                               ; preds = %75
  %25 = load i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !6
  br label %26

26:                                               ; preds = %0, %24
  %27 = phi i16 [ %25, %24 ], [ 0, %0 ]
  %28 = phi i64 [ %116, %24 ], [ 0, %0 ]
  %29 = phi i32 [ %115, %24 ], [ 0, %0 ]
  %30 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %28
  %31 = load i32, ptr %30, align 4, !tbaa !10
  %32 = trunc i32 %31 to i16
  %33 = icmp eq i16 %27, 0
  br i1 %33, label %34, label %75

34:                                               ; preds = %26
  store <8 x i16> %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), align 2, !tbaa !6
  store <8 x i16> %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 18), align 2, !tbaa !6
  store <8 x i16> %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 34), align 2, !tbaa !6
  store <8 x i16> %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 50), align 2, !tbaa !6
  store <8 x i16> %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 66), align 2, !tbaa !6
  store <8 x i16> %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 82), align 2, !tbaa !6
  store <8 x i16> %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 98), align 2, !tbaa !6
  store <4 x i16> %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 114), align 2, !tbaa !6
  store i16 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 122), align 2, !tbaa !6
  store i16 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 124), align 4, !tbaa !6
  store i16 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 126), align 2, !tbaa !6
  %35 = load <8 x i16>, ptr @CRCTable, align 16, !tbaa !6
  %36 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !6
  %37 = xor <8 x i16> %35, splat (i16 18628)
  %38 = xor <8 x i16> %36, splat (i16 18628)
  store <8 x i16> %37, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 16, !tbaa !6
  store <8 x i16> %38, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !6
  %39 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !6
  %40 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !6
  %41 = xor <8 x i16> %39, splat (i16 18628)
  %42 = xor <8 x i16> %40, splat (i16 18628)
  store <8 x i16> %41, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 16, !tbaa !6
  store <8 x i16> %42, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !6
  %43 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !6
  %44 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !6
  %45 = xor <8 x i16> %43, splat (i16 18628)
  %46 = xor <8 x i16> %44, splat (i16 18628)
  store <8 x i16> %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 16, !tbaa !6
  store <8 x i16> %46, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !6
  %47 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !6
  %48 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !6
  %49 = xor <8 x i16> %47, splat (i16 18628)
  %50 = xor <8 x i16> %48, splat (i16 18628)
  store <8 x i16> %49, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 16, !tbaa !6
  store <8 x i16> %50, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !6
  %51 = load <8 x i16>, ptr @CRCTable, align 16, !tbaa !6
  %52 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !6
  %53 = xor <8 x i16> %51, splat (i16 -28280)
  %54 = xor <8 x i16> %52, splat (i16 -28280)
  store <8 x i16> %53, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 16, !tbaa !6
  store <8 x i16> %54, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 272), align 16, !tbaa !6
  %55 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !6
  %56 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !6
  %57 = xor <8 x i16> %55, splat (i16 -28280)
  %58 = xor <8 x i16> %56, splat (i16 -28280)
  store <8 x i16> %57, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 288), align 16, !tbaa !6
  store <8 x i16> %58, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 304), align 16, !tbaa !6
  %59 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !6
  %60 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !6
  %61 = xor <8 x i16> %59, splat (i16 -28280)
  %62 = xor <8 x i16> %60, splat (i16 -28280)
  store <8 x i16> %61, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 16, !tbaa !6
  store <8 x i16> %62, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 336), align 16, !tbaa !6
  %63 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !6
  %64 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !6
  %65 = xor <8 x i16> %63, splat (i16 -28280)
  %66 = xor <8 x i16> %64, splat (i16 -28280)
  store <8 x i16> %65, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 352), align 16, !tbaa !6
  store <8 x i16> %66, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 368), align 16, !tbaa !6
  %67 = xor <8 x i16> %37, splat (i16 -28280)
  %68 = xor <8 x i16> %38, splat (i16 -28280)
  store <8 x i16> %67, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 16, !tbaa !6
  store <8 x i16> %68, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 400), align 16, !tbaa !6
  %69 = xor <8 x i16> %41, splat (i16 -28280)
  %70 = xor <8 x i16> %42, splat (i16 -28280)
  store <8 x i16> %69, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 416), align 16, !tbaa !6
  store <8 x i16> %70, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 432), align 16, !tbaa !6
  %71 = xor <8 x i16> %45, splat (i16 -28280)
  %72 = xor <8 x i16> %46, splat (i16 -28280)
  store <8 x i16> %71, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 448), align 16, !tbaa !6
  store <8 x i16> %72, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 464), align 16, !tbaa !6
  %73 = xor <8 x i16> %49, splat (i16 -28280)
  %74 = xor <8 x i16> %50, splat (i16 -28280)
  store <8 x i16> %73, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 480), align 16, !tbaa !6
  store <8 x i16> %74, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 496), align 16, !tbaa !6
  br label %75

75:                                               ; preds = %34, %26
  %76 = lshr i16 %32, 8
  %77 = zext nneg i16 %76 to i64
  %78 = shl i16 %32, 8
  %79 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %77
  %80 = load i16, ptr %79, align 2, !tbaa !6
  %81 = xor i16 %80, %78
  %82 = shl i16 %32, 1
  %83 = xor i16 %82, 4129
  %84 = icmp slt i16 %32, 0
  %85 = select i1 %84, i16 %83, i16 %82
  %86 = shl i16 %85, 1
  %87 = xor i16 %86, 4129
  %88 = icmp slt i16 %85, 0
  %89 = select i1 %88, i16 %87, i16 %86
  %90 = shl i16 %89, 1
  %91 = xor i16 %90, 4129
  %92 = icmp slt i16 %89, 0
  %93 = select i1 %92, i16 %91, i16 %90
  %94 = shl i16 %93, 1
  %95 = xor i16 %94, 4129
  %96 = icmp slt i16 %93, 0
  %97 = select i1 %96, i16 %95, i16 %94
  %98 = shl i16 %97, 1
  %99 = xor i16 %98, 4129
  %100 = icmp slt i16 %97, 0
  %101 = select i1 %100, i16 %99, i16 %98
  %102 = shl i16 %101, 1
  %103 = xor i16 %102, 4129
  %104 = icmp slt i16 %101, 0
  %105 = select i1 %104, i16 %103, i16 %102
  %106 = shl i16 %105, 1
  %107 = xor i16 %106, 4129
  %108 = icmp slt i16 %105, 0
  %109 = select i1 %108, i16 %107, i16 %106
  %110 = shl i16 %109, 1
  %111 = xor i16 %110, 4129
  %112 = icmp slt i16 %109, 0
  %113 = select i1 %112, i16 %111, i16 %110
  %114 = icmp eq i16 %81, %113
  %115 = select i1 %114, i32 %29, i32 1
  %116 = add nuw nsw i64 %28, 1
  %117 = icmp eq i64 %116, 8
  br i1 %117, label %22, label %24, !llvm.loop !12
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
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13, !14}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.unswitch.partial.disable"}
