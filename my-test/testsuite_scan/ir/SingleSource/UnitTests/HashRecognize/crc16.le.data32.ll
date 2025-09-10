; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.le.data32.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.le.data32.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.sample = internal unnamed_addr constant [8 x i32] [i32 0, i32 1, i32 11, i32 16, i32 129, i32 142, i32 196, i32 255], align 4
@CRCTable = internal unnamed_addr global [256 x i16] zeroinitializer, align 64

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  %1 = load i16, ptr @CRCTable, align 64
  %2 = xor i16 %1, -24575
  %3 = xor i16 %1, -4095
  %4 = xor i16 %1, 20480
  %5 = xor i16 %1, -10239
  %6 = xor i16 %1, 10240
  %7 = xor i16 %1, 30720
  %8 = xor i16 %1, -30719
  %9 = xor i16 %1, -13311
  %10 = xor i16 %1, 5120
  %11 = xor i16 %1, 15360
  %12 = xor i16 %1, -7167
  %13 = xor i16 %1, 27648
  %14 = xor i16 %1, -19455
  %15 = xor i16 %1, -25599
  %16 = xor i16 %1, 17408
  %17 = xor i16 %1, -14847
  %18 = xor i16 %1, 2560
  %19 = xor i16 %1, 7680
  %20 = xor i16 %1, -11775
  %21 = xor i16 %1, 13824
  %22 = xor i16 %1, -1535
  %23 = xor i16 %1, -4607
  %24 = xor i16 %1, 8704
  %25 = xor i16 %1, 26112
  %26 = xor i16 %1, -22015
  %27 = xor i16 %1, -16895
  %28 = xor i16 %1, 29184
  %29 = xor i16 %1, -27135
  %30 = xor i16 %1, 23040
  %31 = xor i16 %1, 19968
  %32 = xor i16 %1, -32255
  %33 = xor i16 %1, -15615
  %34 = xor i16 %1, 1280
  %35 = xor i16 %1, 3840
  %36 = xor i16 %1, -14079
  %37 = xor i16 %1, 6912
  %38 = xor i16 %1, -8959
  %39 = xor i16 %1, -10495
  %40 = xor i16 %1, 4352
  %41 = xor i16 %1, 13056
  %42 = xor i16 %1, -2815
  %43 = xor i16 %1, -255
  %44 = xor i16 %1, 14592
  %45 = xor i16 %1, -5375
  %46 = xor i16 %1, 11520
  %47 = xor i16 %1, 9984
  %48 = xor i16 %1, -7935
  %49 = xor i16 %1, 25344
  %50 = xor i16 %1, -23295
  %51 = xor i16 %1, -20735
  %52 = xor i16 %1, 26880
  %53 = xor i16 %1, -17663
  %54 = xor i16 %1, 32000
  %55 = xor i16 %1, 30464
  %56 = xor i16 %1, -20223
  %57 = xor i16 %1, -27903
  %58 = xor i16 %1, 21760
  %59 = xor i16 %1, 24320
  %60 = xor i16 %1, -26367
  %61 = xor i16 %1, 19200
  %62 = xor i16 %1, -29439
  %63 = xor i16 %1, -30975
  %64 = xor i16 %1, 16640
  br label %66

65:                                               ; preds = %204
  ret i32 %475

66:                                               ; preds = %0, %204
  %67 = phi i64 [ 0, %0 ], [ %476, %204 ]
  %68 = phi i32 [ 0, %0 ], [ %475, %204 ]
  %69 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %67
  %70 = load i32, ptr %69, align 4, !tbaa !6
  %71 = trunc i32 %70 to i16
  %72 = sub nuw nsw i64 7, %67
  %73 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %72
  %74 = load i32, ptr %73, align 4, !tbaa !6
  %75 = load i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !10
  %76 = icmp eq i16 %75, 0
  br i1 %76, label %77, label %204

77:                                               ; preds = %66
  store i16 %2, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 64, !tbaa !10
  store i16 %3, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 64, !tbaa !10
  store i16 %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 64, !tbaa !10
  store i16 %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 64, !tbaa !10
  store i16 %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 64, !tbaa !10
  store i16 %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 64, !tbaa !10
  store i16 %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 448), align 64, !tbaa !10
  store i16 %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 32, !tbaa !10
  store i16 %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 32, !tbaa !10
  store i16 %11, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 32, !tbaa !10
  store i16 %12, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 32, !tbaa !10
  store i16 %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 288), align 32, !tbaa !10
  store i16 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 352), align 32, !tbaa !10
  store i16 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 416), align 32, !tbaa !10
  store i16 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 480), align 32, !tbaa !10
  store i16 %17, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !10
  store i16 %18, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !10
  store i16 %19, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !10
  store i16 %20, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !10
  store i16 %21, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !10
  store i16 %22, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !10
  store i16 %23, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !10
  store i16 %24, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !10
  store i16 %25, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 272), align 16, !tbaa !10
  store i16 %26, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 304), align 16, !tbaa !10
  store i16 %27, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 336), align 16, !tbaa !10
  store i16 %28, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 368), align 16, !tbaa !10
  store i16 %29, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 400), align 16, !tbaa !10
  store i16 %30, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 432), align 16, !tbaa !10
  store i16 %31, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 464), align 16, !tbaa !10
  store i16 %32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 496), align 16, !tbaa !10
  store i16 %33, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 8), align 8, !tbaa !10
  store i16 %34, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 24), align 8, !tbaa !10
  store i16 %35, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 40), align 8, !tbaa !10
  store i16 %36, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 56), align 8, !tbaa !10
  store i16 %37, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 72), align 8, !tbaa !10
  store i16 %38, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 88), align 8, !tbaa !10
  store i16 %39, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 104), align 8, !tbaa !10
  store i16 %40, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 120), align 8, !tbaa !10
  store i16 %41, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 136), align 8, !tbaa !10
  store i16 %42, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 152), align 8, !tbaa !10
  store i16 %43, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 168), align 8, !tbaa !10
  store i16 %44, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 184), align 8, !tbaa !10
  store i16 %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 200), align 8, !tbaa !10
  store i16 %46, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 216), align 8, !tbaa !10
  store i16 %47, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 232), align 8, !tbaa !10
  store i16 %48, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 248), align 8, !tbaa !10
  store i16 %49, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 264), align 8, !tbaa !10
  store i16 %50, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 280), align 8, !tbaa !10
  store i16 %51, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 296), align 8, !tbaa !10
  store i16 %52, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 312), align 8, !tbaa !10
  store i16 %53, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 328), align 8, !tbaa !10
  store i16 %54, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 344), align 8, !tbaa !10
  store i16 %55, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 360), align 8, !tbaa !10
  store i16 %56, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 376), align 8, !tbaa !10
  store i16 %57, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 392), align 8, !tbaa !10
  store i16 %58, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 408), align 8, !tbaa !10
  store i16 %59, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 424), align 8, !tbaa !10
  store i16 %60, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 440), align 8, !tbaa !10
  store i16 %61, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 456), align 8, !tbaa !10
  store i16 %62, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 472), align 8, !tbaa !10
  store i16 %63, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 488), align 8, !tbaa !10
  store i16 %64, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 504), align 8, !tbaa !10
  %78 = load <29 x i16>, ptr @CRCTable, align 64, !tbaa !10
  %79 = shufflevector <29 x i16> %78, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %80 = xor <8 x i16> %79, splat (i16 -15999)
  %81 = extractelement <8 x i16> %80, i64 0
  store i16 %81, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 4), align 4, !tbaa !10
  %82 = extractelement <8 x i16> %80, i64 1
  store i16 %82, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 12), align 4, !tbaa !10
  %83 = extractelement <8 x i16> %80, i64 2
  store i16 %83, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 20), align 4, !tbaa !10
  %84 = extractelement <8 x i16> %80, i64 3
  store i16 %84, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 28), align 4, !tbaa !10
  %85 = extractelement <8 x i16> %80, i64 4
  store i16 %85, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 36), align 4, !tbaa !10
  %86 = extractelement <8 x i16> %80, i64 5
  store i16 %86, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 44), align 4, !tbaa !10
  %87 = extractelement <8 x i16> %80, i64 6
  store i16 %87, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 52), align 4, !tbaa !10
  %88 = extractelement <8 x i16> %80, i64 7
  store i16 %88, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 60), align 4, !tbaa !10
  %89 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 64, !tbaa !10
  %90 = shufflevector <29 x i16> %89, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %91 = xor <8 x i16> %90, splat (i16 -15999)
  %92 = extractelement <8 x i16> %91, i64 0
  store i16 %92, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 68), align 4, !tbaa !10
  %93 = extractelement <8 x i16> %91, i64 1
  store i16 %93, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 76), align 4, !tbaa !10
  %94 = extractelement <8 x i16> %91, i64 2
  store i16 %94, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 84), align 4, !tbaa !10
  %95 = extractelement <8 x i16> %91, i64 3
  store i16 %95, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 92), align 4, !tbaa !10
  %96 = extractelement <8 x i16> %91, i64 4
  store i16 %96, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 100), align 4, !tbaa !10
  %97 = extractelement <8 x i16> %91, i64 5
  store i16 %97, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 108), align 4, !tbaa !10
  %98 = extractelement <8 x i16> %91, i64 6
  store i16 %98, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 116), align 4, !tbaa !10
  %99 = extractelement <8 x i16> %91, i64 7
  store i16 %99, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 124), align 4, !tbaa !10
  %100 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 64, !tbaa !10
  %101 = shufflevector <29 x i16> %100, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %102 = xor <8 x i16> %101, splat (i16 -15999)
  %103 = extractelement <8 x i16> %102, i64 0
  store i16 %103, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 132), align 4, !tbaa !10
  %104 = extractelement <8 x i16> %102, i64 1
  store i16 %104, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 140), align 4, !tbaa !10
  %105 = extractelement <8 x i16> %102, i64 2
  store i16 %105, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 148), align 4, !tbaa !10
  %106 = extractelement <8 x i16> %102, i64 3
  store i16 %106, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 156), align 4, !tbaa !10
  %107 = extractelement <8 x i16> %102, i64 4
  store i16 %107, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 164), align 4, !tbaa !10
  %108 = extractelement <8 x i16> %102, i64 5
  store i16 %108, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 172), align 4, !tbaa !10
  %109 = extractelement <8 x i16> %102, i64 6
  store i16 %109, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 180), align 4, !tbaa !10
  %110 = extractelement <8 x i16> %102, i64 7
  store i16 %110, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 188), align 4, !tbaa !10
  %111 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 64, !tbaa !10
  %112 = shufflevector <29 x i16> %111, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %113 = xor <8 x i16> %112, splat (i16 -15999)
  %114 = extractelement <8 x i16> %113, i64 0
  store i16 %114, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 196), align 4, !tbaa !10
  %115 = extractelement <8 x i16> %113, i64 1
  store i16 %115, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 204), align 4, !tbaa !10
  %116 = extractelement <8 x i16> %113, i64 2
  store i16 %116, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 212), align 4, !tbaa !10
  %117 = extractelement <8 x i16> %113, i64 3
  store i16 %117, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 220), align 4, !tbaa !10
  %118 = extractelement <8 x i16> %113, i64 4
  store i16 %118, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 228), align 4, !tbaa !10
  %119 = extractelement <8 x i16> %113, i64 5
  store i16 %119, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 236), align 4, !tbaa !10
  %120 = extractelement <8 x i16> %113, i64 6
  store i16 %120, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 244), align 4, !tbaa !10
  %121 = extractelement <8 x i16> %113, i64 7
  store i16 %121, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 252), align 4, !tbaa !10
  %122 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 64, !tbaa !10
  %123 = shufflevector <29 x i16> %122, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %124 = xor <8 x i16> %123, splat (i16 -15999)
  %125 = extractelement <8 x i16> %124, i64 0
  store i16 %125, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 260), align 4, !tbaa !10
  %126 = extractelement <8 x i16> %124, i64 1
  store i16 %126, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 268), align 4, !tbaa !10
  %127 = extractelement <8 x i16> %124, i64 2
  store i16 %127, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 276), align 4, !tbaa !10
  %128 = extractelement <8 x i16> %124, i64 3
  store i16 %128, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 284), align 4, !tbaa !10
  %129 = extractelement <8 x i16> %124, i64 4
  store i16 %129, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 292), align 4, !tbaa !10
  %130 = extractelement <8 x i16> %124, i64 5
  store i16 %130, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 300), align 4, !tbaa !10
  %131 = extractelement <8 x i16> %124, i64 6
  store i16 %131, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 308), align 4, !tbaa !10
  %132 = extractelement <8 x i16> %124, i64 7
  store i16 %132, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 316), align 4, !tbaa !10
  %133 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 64, !tbaa !10
  %134 = shufflevector <29 x i16> %133, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %135 = xor <8 x i16> %134, splat (i16 -15999)
  %136 = extractelement <8 x i16> %135, i64 0
  store i16 %136, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 324), align 4, !tbaa !10
  %137 = extractelement <8 x i16> %135, i64 1
  store i16 %137, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 332), align 4, !tbaa !10
  %138 = extractelement <8 x i16> %135, i64 2
  store i16 %138, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 340), align 4, !tbaa !10
  %139 = extractelement <8 x i16> %135, i64 3
  store i16 %139, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 348), align 4, !tbaa !10
  %140 = extractelement <8 x i16> %135, i64 4
  store i16 %140, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 356), align 4, !tbaa !10
  %141 = extractelement <8 x i16> %135, i64 5
  store i16 %141, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 364), align 4, !tbaa !10
  %142 = extractelement <8 x i16> %135, i64 6
  store i16 %142, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 372), align 4, !tbaa !10
  %143 = extractelement <8 x i16> %135, i64 7
  store i16 %143, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 380), align 4, !tbaa !10
  %144 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 64, !tbaa !10
  %145 = shufflevector <29 x i16> %144, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %146 = xor <8 x i16> %145, splat (i16 -15999)
  %147 = extractelement <8 x i16> %146, i64 0
  store i16 %147, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 388), align 4, !tbaa !10
  %148 = extractelement <8 x i16> %146, i64 1
  store i16 %148, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 396), align 4, !tbaa !10
  %149 = extractelement <8 x i16> %146, i64 2
  store i16 %149, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 404), align 4, !tbaa !10
  %150 = extractelement <8 x i16> %146, i64 3
  store i16 %150, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 412), align 4, !tbaa !10
  %151 = extractelement <8 x i16> %146, i64 4
  store i16 %151, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 420), align 4, !tbaa !10
  %152 = extractelement <8 x i16> %146, i64 5
  store i16 %152, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 428), align 4, !tbaa !10
  %153 = extractelement <8 x i16> %146, i64 6
  store i16 %153, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 436), align 4, !tbaa !10
  %154 = extractelement <8 x i16> %146, i64 7
  store i16 %154, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 444), align 4, !tbaa !10
  %155 = xor i16 %8, -15999
  store i16 %155, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 452), align 4, !tbaa !10
  %156 = xor i16 %61, -15999
  store i16 %156, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 460), align 4, !tbaa !10
  %157 = xor i16 %31, -15999
  store i16 %157, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 468), align 4, !tbaa !10
  %158 = xor i16 %62, -15999
  store i16 %158, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 476), align 4, !tbaa !10
  %159 = xor i16 %16, -15999
  store i16 %159, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 484), align 4, !tbaa !10
  %160 = xor i16 %63, -15999
  store i16 %160, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 492), align 4, !tbaa !10
  %161 = xor i16 %32, -15999
  store i16 %161, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 500), align 4, !tbaa !10
  %162 = xor i16 %64, -15999
  store i16 %162, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 508), align 4, !tbaa !10
  br label %163

163:                                              ; preds = %77, %163
  %164 = phi i64 [ %193, %163 ], [ 0, %77 ]
  %165 = shl i64 %164, 1
  %166 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %165
  %167 = load <15 x i16>, ptr %166, align 32, !tbaa !10
  %168 = shufflevector <15 x i16> %167, <15 x i16> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %169 = xor <8 x i16> %168, splat (i16 -16191)
  %170 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %165
  %171 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %165
  %172 = getelementptr inbounds nuw i8, ptr %171, i64 4
  %173 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %165
  %174 = getelementptr inbounds nuw i8, ptr %173, i64 8
  %175 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %165
  %176 = getelementptr inbounds nuw i8, ptr %175, i64 12
  %177 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %165
  %178 = getelementptr inbounds nuw i8, ptr %177, i64 16
  %179 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %165
  %180 = getelementptr inbounds nuw i8, ptr %179, i64 20
  %181 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %165
  %182 = getelementptr inbounds nuw i8, ptr %181, i64 24
  %183 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %165
  %184 = getelementptr inbounds nuw i8, ptr %183, i64 28
  %185 = extractelement <8 x i16> %169, i64 0
  store i16 %185, ptr %170, align 2, !tbaa !10
  %186 = extractelement <8 x i16> %169, i64 1
  store i16 %186, ptr %172, align 2, !tbaa !10
  %187 = extractelement <8 x i16> %169, i64 2
  store i16 %187, ptr %174, align 2, !tbaa !10
  %188 = extractelement <8 x i16> %169, i64 3
  store i16 %188, ptr %176, align 2, !tbaa !10
  %189 = extractelement <8 x i16> %169, i64 4
  store i16 %189, ptr %178, align 2, !tbaa !10
  %190 = extractelement <8 x i16> %169, i64 5
  store i16 %190, ptr %180, align 2, !tbaa !10
  %191 = extractelement <8 x i16> %169, i64 6
  store i16 %191, ptr %182, align 2, !tbaa !10
  %192 = extractelement <8 x i16> %169, i64 7
  store i16 %192, ptr %184, align 2, !tbaa !10
  %193 = add nuw i64 %164, 8
  %194 = icmp eq i64 %193, 120
  br i1 %194, label %195, label %163, !llvm.loop !12

195:                                              ; preds = %163
  %196 = xor i16 %16, -16191
  store i16 %196, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 482), align 2, !tbaa !10
  %197 = xor i16 %159, -16191
  store i16 %197, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 486), align 2, !tbaa !10
  %198 = xor i16 %63, -16191
  store i16 %198, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 490), align 2, !tbaa !10
  %199 = xor i16 %160, -16191
  store i16 %199, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 494), align 2, !tbaa !10
  %200 = xor i16 %32, -16191
  store i16 %200, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 498), align 2, !tbaa !10
  %201 = xor i16 %161, -16191
  store i16 %201, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 502), align 2, !tbaa !10
  %202 = xor i16 %64, -16191
  store i16 %202, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 506), align 2, !tbaa !10
  %203 = xor i16 %162, -16191
  store i16 %203, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !10
  br label %204

204:                                              ; preds = %195, %66
  %205 = xor i32 %74, %70
  %206 = lshr i16 %71, 8
  %207 = and i32 %205, 255
  %208 = zext nneg i32 %207 to i64
  %209 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %208
  %210 = load i16, ptr %209, align 2, !tbaa !10
  %211 = xor i16 %210, %206
  %212 = zext i16 %211 to i32
  %213 = lshr i32 %74, 8
  %214 = xor i32 %213, %212
  %215 = lshr i16 %210, 8
  %216 = and i32 %214, 255
  %217 = zext nneg i32 %216 to i64
  %218 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %217
  %219 = load i16, ptr %218, align 2, !tbaa !10
  %220 = xor i16 %219, %215
  %221 = zext i16 %220 to i32
  %222 = lshr i32 %74, 16
  %223 = xor i32 %222, %221
  %224 = lshr i16 %219, 8
  %225 = and i32 %223, 255
  %226 = zext nneg i32 %225 to i64
  %227 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %226
  %228 = load i16, ptr %227, align 2, !tbaa !10
  %229 = lshr i32 %74, 24
  %230 = lshr i16 %228, 8
  %231 = and i16 %228, 255
  %232 = xor i16 %231, %224
  %233 = zext nneg i16 %232 to i32
  %234 = xor i32 %229, %233
  %235 = zext nneg i32 %234 to i64
  %236 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %235
  %237 = load i16, ptr %236, align 2, !tbaa !10
  %238 = xor i16 %237, %230
  %239 = lshr i16 %71, 1
  %240 = and i32 %205, 1
  %241 = icmp eq i32 %240, 0
  %242 = xor i16 %239, -24575
  %243 = select i1 %241, i16 %239, i16 %242
  %244 = trunc i32 %74 to i16
  %245 = lshr i16 %244, 1
  %246 = xor i16 %243, %245
  %247 = lshr i16 %243, 1
  %248 = and i16 %246, 1
  %249 = icmp eq i16 %248, 0
  %250 = xor i16 %247, -24575
  %251 = select i1 %249, i16 %247, i16 %250
  %252 = lshr i16 %244, 2
  %253 = xor i16 %251, %252
  %254 = lshr i16 %251, 1
  %255 = and i16 %253, 1
  %256 = icmp eq i16 %255, 0
  %257 = xor i16 %254, -24575
  %258 = select i1 %256, i16 %254, i16 %257
  %259 = lshr i16 %244, 3
  %260 = xor i16 %258, %259
  %261 = lshr i16 %258, 1
  %262 = and i16 %260, 1
  %263 = icmp eq i16 %262, 0
  %264 = xor i16 %261, -24575
  %265 = select i1 %263, i16 %261, i16 %264
  %266 = lshr i16 %244, 4
  %267 = xor i16 %265, %266
  %268 = lshr i16 %265, 1
  %269 = and i16 %267, 1
  %270 = icmp eq i16 %269, 0
  %271 = xor i16 %268, -24575
  %272 = select i1 %270, i16 %268, i16 %271
  %273 = lshr i16 %244, 5
  %274 = xor i16 %272, %273
  %275 = lshr i16 %272, 1
  %276 = and i16 %274, 1
  %277 = icmp eq i16 %276, 0
  %278 = xor i16 %275, -24575
  %279 = select i1 %277, i16 %275, i16 %278
  %280 = lshr i16 %244, 6
  %281 = xor i16 %279, %280
  %282 = lshr i16 %279, 1
  %283 = and i16 %281, 1
  %284 = icmp eq i16 %283, 0
  %285 = xor i16 %282, -24575
  %286 = select i1 %284, i16 %282, i16 %285
  %287 = lshr i16 %244, 7
  %288 = xor i16 %286, %287
  %289 = lshr i16 %286, 1
  %290 = and i16 %288, 1
  %291 = icmp eq i16 %290, 0
  %292 = xor i16 %289, -24575
  %293 = select i1 %291, i16 %289, i16 %292
  %294 = trunc i32 %213 to i16
  %295 = xor i16 %293, %294
  %296 = lshr i16 %293, 1
  %297 = and i16 %295, 1
  %298 = icmp eq i16 %297, 0
  %299 = xor i16 %296, -24575
  %300 = select i1 %298, i16 %296, i16 %299
  %301 = lshr i16 %244, 9
  %302 = xor i16 %300, %301
  %303 = lshr i16 %300, 1
  %304 = and i16 %302, 1
  %305 = icmp eq i16 %304, 0
  %306 = xor i16 %303, -24575
  %307 = select i1 %305, i16 %303, i16 %306
  %308 = lshr i16 %244, 10
  %309 = xor i16 %307, %308
  %310 = lshr i16 %307, 1
  %311 = and i16 %309, 1
  %312 = icmp eq i16 %311, 0
  %313 = xor i16 %310, -24575
  %314 = select i1 %312, i16 %310, i16 %313
  %315 = lshr i16 %244, 11
  %316 = xor i16 %314, %315
  %317 = lshr i16 %314, 1
  %318 = and i16 %316, 1
  %319 = icmp eq i16 %318, 0
  %320 = xor i16 %317, -24575
  %321 = select i1 %319, i16 %317, i16 %320
  %322 = lshr i16 %244, 12
  %323 = xor i16 %321, %322
  %324 = lshr i16 %321, 1
  %325 = and i16 %323, 1
  %326 = icmp eq i16 %325, 0
  %327 = xor i16 %324, -24575
  %328 = select i1 %326, i16 %324, i16 %327
  %329 = lshr i16 %244, 13
  %330 = xor i16 %328, %329
  %331 = lshr i16 %328, 1
  %332 = and i16 %330, 1
  %333 = icmp eq i16 %332, 0
  %334 = xor i16 %331, -24575
  %335 = select i1 %333, i16 %331, i16 %334
  %336 = lshr i16 %244, 14
  %337 = xor i16 %335, %336
  %338 = lshr i16 %335, 1
  %339 = and i16 %337, 1
  %340 = icmp eq i16 %339, 0
  %341 = xor i16 %338, -24575
  %342 = select i1 %340, i16 %338, i16 %341
  %343 = lshr i16 %244, 15
  %344 = lshr i16 %342, 1
  %345 = and i16 %342, 1
  %346 = icmp eq i16 %345, %343
  %347 = xor i16 %344, -24575
  %348 = select i1 %346, i16 %344, i16 %347
  %349 = trunc nuw i32 %222 to i16
  %350 = xor i16 %348, %349
  %351 = lshr i16 %348, 1
  %352 = and i16 %350, 1
  %353 = icmp eq i16 %352, 0
  %354 = xor i16 %351, -24575
  %355 = select i1 %353, i16 %351, i16 %354
  %356 = lshr i32 %74, 17
  %357 = trunc nuw nsw i32 %356 to i16
  %358 = xor i16 %355, %357
  %359 = lshr i16 %355, 1
  %360 = and i16 %358, 1
  %361 = icmp eq i16 %360, 0
  %362 = xor i16 %359, -24575
  %363 = select i1 %361, i16 %359, i16 %362
  %364 = lshr i32 %74, 18
  %365 = trunc nuw nsw i32 %364 to i16
  %366 = xor i16 %363, %365
  %367 = lshr i16 %363, 1
  %368 = and i16 %366, 1
  %369 = icmp eq i16 %368, 0
  %370 = xor i16 %367, -24575
  %371 = select i1 %369, i16 %367, i16 %370
  %372 = lshr i32 %74, 19
  %373 = trunc nuw nsw i32 %372 to i16
  %374 = xor i16 %371, %373
  %375 = lshr i16 %371, 1
  %376 = and i16 %374, 1
  %377 = icmp eq i16 %376, 0
  %378 = xor i16 %375, -24575
  %379 = select i1 %377, i16 %375, i16 %378
  %380 = lshr i32 %74, 20
  %381 = trunc nuw nsw i32 %380 to i16
  %382 = xor i16 %379, %381
  %383 = lshr i16 %379, 1
  %384 = and i16 %382, 1
  %385 = icmp eq i16 %384, 0
  %386 = xor i16 %383, -24575
  %387 = select i1 %385, i16 %383, i16 %386
  %388 = lshr i32 %74, 21
  %389 = trunc nuw nsw i32 %388 to i16
  %390 = xor i16 %387, %389
  %391 = lshr i16 %387, 1
  %392 = and i16 %390, 1
  %393 = icmp eq i16 %392, 0
  %394 = xor i16 %391, -24575
  %395 = select i1 %393, i16 %391, i16 %394
  %396 = lshr i32 %74, 22
  %397 = trunc nuw nsw i32 %396 to i16
  %398 = xor i16 %395, %397
  %399 = lshr i16 %395, 1
  %400 = and i16 %398, 1
  %401 = icmp eq i16 %400, 0
  %402 = xor i16 %399, -24575
  %403 = select i1 %401, i16 %399, i16 %402
  %404 = lshr i32 %74, 23
  %405 = trunc nuw nsw i32 %404 to i16
  %406 = xor i16 %403, %405
  %407 = lshr i16 %403, 1
  %408 = and i16 %406, 1
  %409 = icmp eq i16 %408, 0
  %410 = xor i16 %407, -24575
  %411 = select i1 %409, i16 %407, i16 %410
  %412 = trunc nuw nsw i32 %229 to i16
  %413 = xor i16 %411, %412
  %414 = lshr i16 %411, 1
  %415 = and i16 %413, 1
  %416 = icmp eq i16 %415, 0
  %417 = xor i16 %414, -24575
  %418 = select i1 %416, i16 %414, i16 %417
  %419 = lshr i32 %74, 25
  %420 = trunc nuw nsw i32 %419 to i16
  %421 = xor i16 %418, %420
  %422 = lshr i16 %418, 1
  %423 = and i16 %421, 1
  %424 = icmp eq i16 %423, 0
  %425 = xor i16 %422, -24575
  %426 = select i1 %424, i16 %422, i16 %425
  %427 = lshr i32 %74, 26
  %428 = trunc nuw nsw i32 %427 to i16
  %429 = xor i16 %426, %428
  %430 = lshr i16 %426, 1
  %431 = and i16 %429, 1
  %432 = icmp eq i16 %431, 0
  %433 = xor i16 %430, -24575
  %434 = select i1 %432, i16 %430, i16 %433
  %435 = lshr i32 %74, 27
  %436 = trunc nuw nsw i32 %435 to i16
  %437 = xor i16 %434, %436
  %438 = lshr i16 %434, 1
  %439 = and i16 %437, 1
  %440 = icmp eq i16 %439, 0
  %441 = xor i16 %438, -24575
  %442 = select i1 %440, i16 %438, i16 %441
  %443 = lshr i32 %74, 28
  %444 = trunc nuw nsw i32 %443 to i16
  %445 = xor i16 %442, %444
  %446 = lshr i16 %442, 1
  %447 = and i16 %445, 1
  %448 = icmp eq i16 %447, 0
  %449 = xor i16 %446, -24575
  %450 = select i1 %448, i16 %446, i16 %449
  %451 = lshr i32 %74, 29
  %452 = trunc nuw nsw i32 %451 to i16
  %453 = xor i16 %450, %452
  %454 = lshr i16 %450, 1
  %455 = and i16 %453, 1
  %456 = icmp eq i16 %455, 0
  %457 = xor i16 %454, -24575
  %458 = select i1 %456, i16 %454, i16 %457
  %459 = lshr i32 %74, 30
  %460 = trunc nuw nsw i32 %459 to i16
  %461 = xor i16 %458, %460
  %462 = lshr i16 %458, 1
  %463 = and i16 %461, 1
  %464 = icmp eq i16 %463, 0
  %465 = xor i16 %462, -24575
  %466 = select i1 %464, i16 %462, i16 %465
  %467 = lshr i32 %74, 31
  %468 = trunc nuw nsw i32 %467 to i16
  %469 = lshr i16 %466, 1
  %470 = and i16 %466, 1
  %471 = icmp eq i16 %470, %468
  %472 = xor i16 %469, -24575
  %473 = select i1 %471, i16 %469, i16 %472
  %474 = icmp eq i16 %238, %473
  %475 = select i1 %474, i32 %68, i32 1
  %476 = add nuw nsw i64 %67, 1
  %477 = icmp eq i64 %476, 8
  br i1 %477, label %65, label %66, !llvm.loop !16
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
!12 = distinct !{!12, !13, !14, !15}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!"llvm.loop.isvectorized", i32 1}
!15 = !{!"llvm.loop.unroll.runtime.disable"}
!16 = distinct !{!16, !13}
