; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.be.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.be.c"
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
  br label %18

17:                                               ; preds = %71
  ret i32 %182

18:                                               ; preds = %0, %71
  %19 = phi i64 [ 0, %0 ], [ %183, %71 ]
  %20 = phi i32 [ 0, %0 ], [ %182, %71 ]
  %21 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %19
  %22 = load i32, ptr %21, align 4, !tbaa !6
  %23 = trunc i32 %22 to i16
  %24 = sub nuw nsw i64 7, %19
  %25 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %24
  %26 = load i32, ptr %25, align 4, !tbaa !6
  %27 = trunc i32 %26 to i16
  %28 = load i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !10
  %29 = icmp eq i16 %28, 0
  br i1 %29, label %30, label %71

30:                                               ; preds = %18
  store <8 x i16> %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), align 2, !tbaa !10
  store <8 x i16> %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 18), align 2, !tbaa !10
  store <8 x i16> %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 34), align 2, !tbaa !10
  store <8 x i16> %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 50), align 2, !tbaa !10
  store <8 x i16> %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 66), align 2, !tbaa !10
  store <8 x i16> %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 82), align 2, !tbaa !10
  store <8 x i16> %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 98), align 2, !tbaa !10
  store <4 x i16> %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 114), align 2, !tbaa !10
  store i16 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 122), align 2, !tbaa !10
  store i16 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 124), align 4, !tbaa !10
  store i16 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 126), align 2, !tbaa !10
  %31 = load <8 x i16>, ptr @CRCTable, align 16, !tbaa !10
  %32 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  %33 = xor <8 x i16> %31, splat (i16 18628)
  %34 = xor <8 x i16> %32, splat (i16 18628)
  store <8 x i16> %33, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 16, !tbaa !10
  store <8 x i16> %34, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !10
  %35 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !10
  %36 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  %37 = xor <8 x i16> %35, splat (i16 18628)
  %38 = xor <8 x i16> %36, splat (i16 18628)
  store <8 x i16> %37, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 16, !tbaa !10
  store <8 x i16> %38, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !10
  %39 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !10
  %40 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !10
  %41 = xor <8 x i16> %39, splat (i16 18628)
  %42 = xor <8 x i16> %40, splat (i16 18628)
  store <8 x i16> %41, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 16, !tbaa !10
  store <8 x i16> %42, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !10
  %43 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !10
  %44 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !10
  %45 = xor <8 x i16> %43, splat (i16 18628)
  %46 = xor <8 x i16> %44, splat (i16 18628)
  store <8 x i16> %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 16, !tbaa !10
  store <8 x i16> %46, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !10
  %47 = load <8 x i16>, ptr @CRCTable, align 16, !tbaa !10
  %48 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  %49 = xor <8 x i16> %47, splat (i16 -28280)
  %50 = xor <8 x i16> %48, splat (i16 -28280)
  store <8 x i16> %49, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 16, !tbaa !10
  store <8 x i16> %50, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 272), align 16, !tbaa !10
  %51 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !10
  %52 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  %53 = xor <8 x i16> %51, splat (i16 -28280)
  %54 = xor <8 x i16> %52, splat (i16 -28280)
  store <8 x i16> %53, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 288), align 16, !tbaa !10
  store <8 x i16> %54, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 304), align 16, !tbaa !10
  %55 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !10
  %56 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !10
  %57 = xor <8 x i16> %55, splat (i16 -28280)
  %58 = xor <8 x i16> %56, splat (i16 -28280)
  store <8 x i16> %57, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 16, !tbaa !10
  store <8 x i16> %58, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 336), align 16, !tbaa !10
  %59 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !10
  %60 = load <8 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !10
  %61 = xor <8 x i16> %59, splat (i16 -28280)
  %62 = xor <8 x i16> %60, splat (i16 -28280)
  store <8 x i16> %61, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 352), align 16, !tbaa !10
  store <8 x i16> %62, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 368), align 16, !tbaa !10
  %63 = xor <8 x i16> %33, splat (i16 -28280)
  %64 = xor <8 x i16> %34, splat (i16 -28280)
  store <8 x i16> %63, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 16, !tbaa !10
  store <8 x i16> %64, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 400), align 16, !tbaa !10
  %65 = xor <8 x i16> %37, splat (i16 -28280)
  %66 = xor <8 x i16> %38, splat (i16 -28280)
  store <8 x i16> %65, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 416), align 16, !tbaa !10
  store <8 x i16> %66, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 432), align 16, !tbaa !10
  %67 = xor <8 x i16> %41, splat (i16 -28280)
  %68 = xor <8 x i16> %42, splat (i16 -28280)
  store <8 x i16> %67, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 448), align 16, !tbaa !10
  store <8 x i16> %68, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 464), align 16, !tbaa !10
  %69 = xor <8 x i16> %45, splat (i16 -28280)
  %70 = xor <8 x i16> %46, splat (i16 -28280)
  store <8 x i16> %69, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 480), align 16, !tbaa !10
  store <8 x i16> %70, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 496), align 16, !tbaa !10
  br label %71

71:                                               ; preds = %30, %18
  %72 = xor i16 %27, %23
  %73 = lshr i16 %72, 8
  %74 = shl i16 %23, 8
  %75 = zext nneg i16 %73 to i64
  %76 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %75
  %77 = load i16, ptr %76, align 2, !tbaa !10
  %78 = xor i16 %77, %74
  %79 = lshr i16 %78, 8
  %80 = shl i16 %77, 8
  %81 = and i16 %27, 255
  %82 = xor i16 %79, %81
  %83 = zext nneg i16 %82 to i64
  %84 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %83
  %85 = load i16, ptr %84, align 2, !tbaa !10
  %86 = xor i16 %85, %80
  %87 = shl i16 %23, 1
  %88 = xor i16 %87, 4129
  %89 = icmp slt i16 %72, 0
  %90 = select i1 %89, i16 %88, i16 %87
  %91 = shl i16 %27, 1
  %92 = xor i16 %90, %91
  %93 = shl i16 %90, 1
  %94 = xor i16 %93, 4129
  %95 = icmp slt i16 %92, 0
  %96 = select i1 %95, i16 %94, i16 %93
  %97 = shl i16 %27, 2
  %98 = xor i16 %96, %97
  %99 = shl i16 %96, 1
  %100 = xor i16 %99, 4129
  %101 = icmp slt i16 %98, 0
  %102 = select i1 %101, i16 %100, i16 %99
  %103 = shl i16 %27, 3
  %104 = xor i16 %102, %103
  %105 = shl i16 %102, 1
  %106 = xor i16 %105, 4129
  %107 = icmp slt i16 %104, 0
  %108 = select i1 %107, i16 %106, i16 %105
  %109 = shl i16 %27, 4
  %110 = xor i16 %108, %109
  %111 = shl i16 %108, 1
  %112 = xor i16 %111, 4129
  %113 = icmp slt i16 %110, 0
  %114 = select i1 %113, i16 %112, i16 %111
  %115 = shl i16 %27, 5
  %116 = xor i16 %114, %115
  %117 = shl i16 %114, 1
  %118 = xor i16 %117, 4129
  %119 = icmp slt i16 %116, 0
  %120 = select i1 %119, i16 %118, i16 %117
  %121 = shl i16 %27, 6
  %122 = xor i16 %120, %121
  %123 = shl i16 %120, 1
  %124 = xor i16 %123, 4129
  %125 = icmp slt i16 %122, 0
  %126 = select i1 %125, i16 %124, i16 %123
  %127 = shl i16 %27, 7
  %128 = xor i16 %126, %127
  %129 = shl i16 %126, 1
  %130 = xor i16 %129, 4129
  %131 = icmp slt i16 %128, 0
  %132 = select i1 %131, i16 %130, i16 %129
  %133 = shl i16 %27, 8
  %134 = xor i16 %132, %133
  %135 = shl i16 %132, 1
  %136 = xor i16 %135, 4129
  %137 = icmp slt i16 %134, 0
  %138 = select i1 %137, i16 %136, i16 %135
  %139 = shl i16 %27, 9
  %140 = xor i16 %138, %139
  %141 = shl i16 %138, 1
  %142 = xor i16 %141, 4129
  %143 = icmp slt i16 %140, 0
  %144 = select i1 %143, i16 %142, i16 %141
  %145 = shl i16 %27, 10
  %146 = xor i16 %144, %145
  %147 = shl i16 %144, 1
  %148 = xor i16 %147, 4129
  %149 = icmp slt i16 %146, 0
  %150 = select i1 %149, i16 %148, i16 %147
  %151 = shl i16 %27, 11
  %152 = xor i16 %150, %151
  %153 = shl i16 %150, 1
  %154 = xor i16 %153, 4129
  %155 = icmp slt i16 %152, 0
  %156 = select i1 %155, i16 %154, i16 %153
  %157 = shl i16 %27, 12
  %158 = xor i16 %156, %157
  %159 = shl i16 %156, 1
  %160 = xor i16 %159, 4129
  %161 = icmp slt i16 %158, 0
  %162 = select i1 %161, i16 %160, i16 %159
  %163 = shl i16 %27, 13
  %164 = xor i16 %162, %163
  %165 = shl i16 %162, 1
  %166 = xor i16 %165, 4129
  %167 = icmp slt i16 %164, 0
  %168 = select i1 %167, i16 %166, i16 %165
  %169 = shl i16 %27, 14
  %170 = xor i16 %168, %169
  %171 = shl i16 %168, 1
  %172 = xor i16 %171, 4129
  %173 = icmp slt i16 %170, 0
  %174 = select i1 %173, i16 %172, i16 %171
  %175 = shl i16 %27, 15
  %176 = xor i16 %174, %175
  %177 = shl i16 %174, 1
  %178 = xor i16 %177, 4129
  %179 = icmp slt i16 %176, 0
  %180 = select i1 %179, i16 %178, i16 %177
  %181 = icmp eq i16 %86, %180
  %182 = select i1 %181, i32 %20, i32 1
  %183 = add nuw nsw i64 %19, 1
  %184 = icmp eq i64 %183, 8
  br i1 %184, label %17, label %18, !llvm.loop !12
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
!10 = !{!11, !11, i64 0}
!11 = !{!"short", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
