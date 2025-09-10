; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.be.nodata.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.be.nodata.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.crc_initval = internal unnamed_addr constant [8 x i16] [i16 0, i16 1, i16 11, i16 16, i16 129, i16 255, i16 4129, i16 16384], align 2
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
  br label %20

17:                                               ; preds = %127
  %18 = add nuw nsw i64 %21, 1
  %19 = icmp eq i64 %18, 8
  br i1 %19, label %141, label %20, !llvm.loop !6

20:                                               ; preds = %0, %17
  %21 = phi i64 [ 0, %0 ], [ %18, %17 ]
  %22 = getelementptr inbounds nuw i16, ptr @main.crc_initval, i64 %21
  %23 = load i16, ptr %22, align 2, !tbaa !8
  %24 = shl i16 %23, 2
  %25 = xor i16 %24, 4129
  %26 = icmp eq i64 %21, 7
  %27 = select i1 %26, i16 %25, i16 %24
  %28 = shl i16 %27, 1
  %29 = xor i16 %28, 4129
  %30 = icmp slt i16 %27, 0
  %31 = select i1 %30, i16 %29, i16 %28
  %32 = shl i16 %31, 1
  %33 = xor i16 %32, 4129
  %34 = icmp slt i16 %31, 0
  %35 = select i1 %34, i16 %33, i16 %32
  %36 = shl i16 %35, 1
  %37 = xor i16 %36, 4129
  %38 = icmp slt i16 %35, 0
  %39 = select i1 %38, i16 %37, i16 %36
  %40 = shl i16 %39, 1
  %41 = xor i16 %40, 4129
  %42 = icmp slt i16 %39, 0
  %43 = select i1 %42, i16 %41, i16 %40
  %44 = shl i16 %43, 1
  %45 = xor i16 %44, 4129
  %46 = icmp slt i16 %43, 0
  %47 = select i1 %46, i16 %45, i16 %44
  %48 = shl i16 %47, 1
  %49 = xor i16 %48, 4129
  %50 = icmp slt i16 %47, 0
  %51 = select i1 %50, i16 %49, i16 %48
  %52 = shl i16 %51, 1
  %53 = xor i16 %52, 4129
  %54 = icmp slt i16 %51, 0
  %55 = select i1 %54, i16 %53, i16 %52
  %56 = shl i16 %55, 1
  %57 = xor i16 %56, 4129
  %58 = icmp slt i16 %55, 0
  %59 = select i1 %58, i16 %57, i16 %56
  %60 = shl i16 %59, 1
  %61 = xor i16 %60, 4129
  %62 = icmp slt i16 %59, 0
  %63 = select i1 %62, i16 %61, i16 %60
  %64 = shl i16 %63, 1
  %65 = xor i16 %64, 4129
  %66 = icmp slt i16 %63, 0
  %67 = select i1 %66, i16 %65, i16 %64
  %68 = shl i16 %67, 1
  %69 = xor i16 %68, 4129
  %70 = icmp slt i16 %67, 0
  %71 = select i1 %70, i16 %69, i16 %68
  %72 = shl i16 %71, 1
  %73 = xor i16 %72, 4129
  %74 = icmp slt i16 %71, 0
  %75 = select i1 %74, i16 %73, i16 %72
  %76 = shl i16 %75, 1
  %77 = xor i16 %76, 4129
  %78 = icmp slt i16 %75, 0
  %79 = select i1 %78, i16 %77, i16 %76
  %80 = shl i16 %79, 1
  %81 = xor i16 %80, 4129
  %82 = icmp slt i16 %79, 0
  %83 = select i1 %82, i16 %81, i16 %80
  %84 = load i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !8
  %85 = icmp eq i16 %84, 0
  br i1 %85, label %86, label %127

86:                                               ; preds = %20
  store <8 x i16> %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), align 2, !tbaa !8
  store <8 x i16> %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 18), align 2, !tbaa !8
  store <8 x i16> %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 34), align 2, !tbaa !8
  store <8 x i16> %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 50), align 2, !tbaa !8
  store <8 x i16> %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 66), align 2, !tbaa !8
  store <8 x i16> %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 82), align 2, !tbaa !8
  store <8 x i16> %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 98), align 2, !tbaa !8
  store <4 x i16> %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 114), align 2, !tbaa !8
  store i16 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 122), align 2, !tbaa !8
  store i16 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 124), align 4, !tbaa !8
  store i16 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 126), align 2, !tbaa !8
  %87 = load <8 x i16>, ptr @CRCTable, align 16, !tbaa !8
  %88 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !8
  %89 = xor <8 x i16> %87, splat (i16 18628)
  %90 = xor <8 x i16> %88, splat (i16 18628)
  store <8 x i16> %89, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 16, !tbaa !8
  store <8 x i16> %90, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !8
  %91 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !8
  %92 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !8
  %93 = xor <8 x i16> %91, splat (i16 18628)
  %94 = xor <8 x i16> %92, splat (i16 18628)
  store <8 x i16> %93, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 16, !tbaa !8
  store <8 x i16> %94, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !8
  %95 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !8
  %96 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !8
  %97 = xor <8 x i16> %95, splat (i16 18628)
  %98 = xor <8 x i16> %96, splat (i16 18628)
  store <8 x i16> %97, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 16, !tbaa !8
  store <8 x i16> %98, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !8
  %99 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !8
  %100 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !8
  %101 = xor <8 x i16> %99, splat (i16 18628)
  %102 = xor <8 x i16> %100, splat (i16 18628)
  store <8 x i16> %101, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 16, !tbaa !8
  store <8 x i16> %102, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !8
  %103 = load <8 x i16>, ptr @CRCTable, align 16, !tbaa !8
  %104 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !8
  %105 = xor <8 x i16> %103, splat (i16 -28280)
  %106 = xor <8 x i16> %104, splat (i16 -28280)
  store <8 x i16> %105, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 16, !tbaa !8
  store <8 x i16> %106, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 272), align 16, !tbaa !8
  %107 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !8
  %108 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !8
  %109 = xor <8 x i16> %107, splat (i16 -28280)
  %110 = xor <8 x i16> %108, splat (i16 -28280)
  store <8 x i16> %109, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 288), align 16, !tbaa !8
  store <8 x i16> %110, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 304), align 16, !tbaa !8
  %111 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !8
  %112 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !8
  %113 = xor <8 x i16> %111, splat (i16 -28280)
  %114 = xor <8 x i16> %112, splat (i16 -28280)
  store <8 x i16> %113, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 16, !tbaa !8
  store <8 x i16> %114, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 336), align 16, !tbaa !8
  %115 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !8
  %116 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !8
  %117 = xor <8 x i16> %115, splat (i16 -28280)
  %118 = xor <8 x i16> %116, splat (i16 -28280)
  store <8 x i16> %117, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 352), align 16, !tbaa !8
  store <8 x i16> %118, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 368), align 16, !tbaa !8
  %119 = xor <8 x i16> %89, splat (i16 -28280)
  %120 = xor <8 x i16> %90, splat (i16 -28280)
  store <8 x i16> %119, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 16, !tbaa !8
  store <8 x i16> %120, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 400), align 16, !tbaa !8
  %121 = xor <8 x i16> %93, splat (i16 -28280)
  %122 = xor <8 x i16> %94, splat (i16 -28280)
  store <8 x i16> %121, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 416), align 16, !tbaa !8
  store <8 x i16> %122, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 432), align 16, !tbaa !8
  %123 = xor <8 x i16> %97, splat (i16 -28280)
  %124 = xor <8 x i16> %98, splat (i16 -28280)
  store <8 x i16> %123, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 448), align 16, !tbaa !8
  store <8 x i16> %124, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 464), align 16, !tbaa !8
  %125 = xor <8 x i16> %101, splat (i16 -28280)
  %126 = xor <8 x i16> %102, splat (i16 -28280)
  store <8 x i16> %125, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 480), align 16, !tbaa !8
  store <8 x i16> %126, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 496), align 16, !tbaa !8
  br label %127

127:                                              ; preds = %86, %20
  %128 = lshr i16 %23, 8
  %129 = zext nneg i16 %128 to i64
  %130 = shl i16 %23, 8
  %131 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %129
  %132 = load i16, ptr %131, align 2, !tbaa !8
  %133 = xor i16 %132, %130
  %134 = lshr i16 %133, 8
  %135 = zext nneg i16 %134 to i64
  %136 = shl i16 %132, 8
  %137 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %135
  %138 = load i16, ptr %137, align 2, !tbaa !8
  %139 = xor i16 %138, %136
  %140 = icmp eq i16 %83, %139
  br i1 %140, label %17, label %141

141:                                              ; preds = %17, %127
  %142 = phi i32 [ 1, %127 ], [ 0, %17 ]
  ret i32 %142
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
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!9, !9, i64 0}
!9 = !{!"short", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
