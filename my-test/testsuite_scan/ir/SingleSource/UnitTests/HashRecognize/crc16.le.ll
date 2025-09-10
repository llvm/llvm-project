; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.le.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.le.c"
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

65:                                               ; preds = %205
  ret i32 %329

66:                                               ; preds = %0, %205
  %67 = phi i64 [ 0, %0 ], [ %330, %205 ]
  %68 = phi i32 [ 0, %0 ], [ %329, %205 ]
  %69 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %67
  %70 = load i32, ptr %69, align 4, !tbaa !6
  %71 = trunc i32 %70 to i16
  %72 = sub nuw nsw i64 7, %67
  %73 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %72
  %74 = load i32, ptr %73, align 4, !tbaa !6
  %75 = trunc i32 %74 to i16
  %76 = load i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !10
  %77 = icmp eq i16 %76, 0
  br i1 %77, label %78, label %205

78:                                               ; preds = %66
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
  %79 = load <29 x i16>, ptr @CRCTable, align 64, !tbaa !10
  %80 = shufflevector <29 x i16> %79, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %81 = xor <8 x i16> %80, splat (i16 -15999)
  %82 = extractelement <8 x i16> %81, i64 0
  store i16 %82, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 4), align 4, !tbaa !10
  %83 = extractelement <8 x i16> %81, i64 1
  store i16 %83, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 12), align 4, !tbaa !10
  %84 = extractelement <8 x i16> %81, i64 2
  store i16 %84, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 20), align 4, !tbaa !10
  %85 = extractelement <8 x i16> %81, i64 3
  store i16 %85, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 28), align 4, !tbaa !10
  %86 = extractelement <8 x i16> %81, i64 4
  store i16 %86, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 36), align 4, !tbaa !10
  %87 = extractelement <8 x i16> %81, i64 5
  store i16 %87, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 44), align 4, !tbaa !10
  %88 = extractelement <8 x i16> %81, i64 6
  store i16 %88, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 52), align 4, !tbaa !10
  %89 = extractelement <8 x i16> %81, i64 7
  store i16 %89, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 60), align 4, !tbaa !10
  %90 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 64, !tbaa !10
  %91 = shufflevector <29 x i16> %90, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %92 = xor <8 x i16> %91, splat (i16 -15999)
  %93 = extractelement <8 x i16> %92, i64 0
  store i16 %93, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 68), align 4, !tbaa !10
  %94 = extractelement <8 x i16> %92, i64 1
  store i16 %94, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 76), align 4, !tbaa !10
  %95 = extractelement <8 x i16> %92, i64 2
  store i16 %95, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 84), align 4, !tbaa !10
  %96 = extractelement <8 x i16> %92, i64 3
  store i16 %96, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 92), align 4, !tbaa !10
  %97 = extractelement <8 x i16> %92, i64 4
  store i16 %97, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 100), align 4, !tbaa !10
  %98 = extractelement <8 x i16> %92, i64 5
  store i16 %98, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 108), align 4, !tbaa !10
  %99 = extractelement <8 x i16> %92, i64 6
  store i16 %99, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 116), align 4, !tbaa !10
  %100 = extractelement <8 x i16> %92, i64 7
  store i16 %100, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 124), align 4, !tbaa !10
  %101 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 64, !tbaa !10
  %102 = shufflevector <29 x i16> %101, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %103 = xor <8 x i16> %102, splat (i16 -15999)
  %104 = extractelement <8 x i16> %103, i64 0
  store i16 %104, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 132), align 4, !tbaa !10
  %105 = extractelement <8 x i16> %103, i64 1
  store i16 %105, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 140), align 4, !tbaa !10
  %106 = extractelement <8 x i16> %103, i64 2
  store i16 %106, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 148), align 4, !tbaa !10
  %107 = extractelement <8 x i16> %103, i64 3
  store i16 %107, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 156), align 4, !tbaa !10
  %108 = extractelement <8 x i16> %103, i64 4
  store i16 %108, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 164), align 4, !tbaa !10
  %109 = extractelement <8 x i16> %103, i64 5
  store i16 %109, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 172), align 4, !tbaa !10
  %110 = extractelement <8 x i16> %103, i64 6
  store i16 %110, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 180), align 4, !tbaa !10
  %111 = extractelement <8 x i16> %103, i64 7
  store i16 %111, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 188), align 4, !tbaa !10
  %112 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 64, !tbaa !10
  %113 = shufflevector <29 x i16> %112, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %114 = xor <8 x i16> %113, splat (i16 -15999)
  %115 = extractelement <8 x i16> %114, i64 0
  store i16 %115, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 196), align 4, !tbaa !10
  %116 = extractelement <8 x i16> %114, i64 1
  store i16 %116, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 204), align 4, !tbaa !10
  %117 = extractelement <8 x i16> %114, i64 2
  store i16 %117, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 212), align 4, !tbaa !10
  %118 = extractelement <8 x i16> %114, i64 3
  store i16 %118, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 220), align 4, !tbaa !10
  %119 = extractelement <8 x i16> %114, i64 4
  store i16 %119, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 228), align 4, !tbaa !10
  %120 = extractelement <8 x i16> %114, i64 5
  store i16 %120, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 236), align 4, !tbaa !10
  %121 = extractelement <8 x i16> %114, i64 6
  store i16 %121, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 244), align 4, !tbaa !10
  %122 = extractelement <8 x i16> %114, i64 7
  store i16 %122, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 252), align 4, !tbaa !10
  %123 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 64, !tbaa !10
  %124 = shufflevector <29 x i16> %123, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %125 = xor <8 x i16> %124, splat (i16 -15999)
  %126 = extractelement <8 x i16> %125, i64 0
  store i16 %126, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 260), align 4, !tbaa !10
  %127 = extractelement <8 x i16> %125, i64 1
  store i16 %127, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 268), align 4, !tbaa !10
  %128 = extractelement <8 x i16> %125, i64 2
  store i16 %128, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 276), align 4, !tbaa !10
  %129 = extractelement <8 x i16> %125, i64 3
  store i16 %129, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 284), align 4, !tbaa !10
  %130 = extractelement <8 x i16> %125, i64 4
  store i16 %130, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 292), align 4, !tbaa !10
  %131 = extractelement <8 x i16> %125, i64 5
  store i16 %131, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 300), align 4, !tbaa !10
  %132 = extractelement <8 x i16> %125, i64 6
  store i16 %132, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 308), align 4, !tbaa !10
  %133 = extractelement <8 x i16> %125, i64 7
  store i16 %133, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 316), align 4, !tbaa !10
  %134 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 64, !tbaa !10
  %135 = shufflevector <29 x i16> %134, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %136 = xor <8 x i16> %135, splat (i16 -15999)
  %137 = extractelement <8 x i16> %136, i64 0
  store i16 %137, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 324), align 4, !tbaa !10
  %138 = extractelement <8 x i16> %136, i64 1
  store i16 %138, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 332), align 4, !tbaa !10
  %139 = extractelement <8 x i16> %136, i64 2
  store i16 %139, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 340), align 4, !tbaa !10
  %140 = extractelement <8 x i16> %136, i64 3
  store i16 %140, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 348), align 4, !tbaa !10
  %141 = extractelement <8 x i16> %136, i64 4
  store i16 %141, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 356), align 4, !tbaa !10
  %142 = extractelement <8 x i16> %136, i64 5
  store i16 %142, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 364), align 4, !tbaa !10
  %143 = extractelement <8 x i16> %136, i64 6
  store i16 %143, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 372), align 4, !tbaa !10
  %144 = extractelement <8 x i16> %136, i64 7
  store i16 %144, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 380), align 4, !tbaa !10
  %145 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 64, !tbaa !10
  %146 = shufflevector <29 x i16> %145, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %147 = xor <8 x i16> %146, splat (i16 -15999)
  %148 = extractelement <8 x i16> %147, i64 0
  store i16 %148, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 388), align 4, !tbaa !10
  %149 = extractelement <8 x i16> %147, i64 1
  store i16 %149, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 396), align 4, !tbaa !10
  %150 = extractelement <8 x i16> %147, i64 2
  store i16 %150, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 404), align 4, !tbaa !10
  %151 = extractelement <8 x i16> %147, i64 3
  store i16 %151, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 412), align 4, !tbaa !10
  %152 = extractelement <8 x i16> %147, i64 4
  store i16 %152, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 420), align 4, !tbaa !10
  %153 = extractelement <8 x i16> %147, i64 5
  store i16 %153, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 428), align 4, !tbaa !10
  %154 = extractelement <8 x i16> %147, i64 6
  store i16 %154, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 436), align 4, !tbaa !10
  %155 = extractelement <8 x i16> %147, i64 7
  store i16 %155, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 444), align 4, !tbaa !10
  %156 = xor i16 %8, -15999
  store i16 %156, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 452), align 4, !tbaa !10
  %157 = xor i16 %61, -15999
  store i16 %157, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 460), align 4, !tbaa !10
  %158 = xor i16 %31, -15999
  store i16 %158, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 468), align 4, !tbaa !10
  %159 = xor i16 %62, -15999
  store i16 %159, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 476), align 4, !tbaa !10
  %160 = xor i16 %16, -15999
  store i16 %160, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 484), align 4, !tbaa !10
  %161 = xor i16 %63, -15999
  store i16 %161, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 492), align 4, !tbaa !10
  %162 = xor i16 %32, -15999
  store i16 %162, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 500), align 4, !tbaa !10
  %163 = xor i16 %64, -15999
  store i16 %163, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 508), align 4, !tbaa !10
  br label %164

164:                                              ; preds = %78, %164
  %165 = phi i64 [ %194, %164 ], [ 0, %78 ]
  %166 = shl i64 %165, 1
  %167 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %166
  %168 = load <15 x i16>, ptr %167, align 32, !tbaa !10
  %169 = shufflevector <15 x i16> %168, <15 x i16> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %170 = xor <8 x i16> %169, splat (i16 -16191)
  %171 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %166
  %172 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %166
  %173 = getelementptr inbounds nuw i8, ptr %172, i64 4
  %174 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %166
  %175 = getelementptr inbounds nuw i8, ptr %174, i64 8
  %176 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %166
  %177 = getelementptr inbounds nuw i8, ptr %176, i64 12
  %178 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %166
  %179 = getelementptr inbounds nuw i8, ptr %178, i64 16
  %180 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %166
  %181 = getelementptr inbounds nuw i8, ptr %180, i64 20
  %182 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %166
  %183 = getelementptr inbounds nuw i8, ptr %182, i64 24
  %184 = getelementptr inbounds nuw i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 2), i64 %166
  %185 = getelementptr inbounds nuw i8, ptr %184, i64 28
  %186 = extractelement <8 x i16> %170, i64 0
  store i16 %186, ptr %171, align 2, !tbaa !10
  %187 = extractelement <8 x i16> %170, i64 1
  store i16 %187, ptr %173, align 2, !tbaa !10
  %188 = extractelement <8 x i16> %170, i64 2
  store i16 %188, ptr %175, align 2, !tbaa !10
  %189 = extractelement <8 x i16> %170, i64 3
  store i16 %189, ptr %177, align 2, !tbaa !10
  %190 = extractelement <8 x i16> %170, i64 4
  store i16 %190, ptr %179, align 2, !tbaa !10
  %191 = extractelement <8 x i16> %170, i64 5
  store i16 %191, ptr %181, align 2, !tbaa !10
  %192 = extractelement <8 x i16> %170, i64 6
  store i16 %192, ptr %183, align 2, !tbaa !10
  %193 = extractelement <8 x i16> %170, i64 7
  store i16 %193, ptr %185, align 2, !tbaa !10
  %194 = add nuw i64 %165, 8
  %195 = icmp eq i64 %194, 120
  br i1 %195, label %196, label %164, !llvm.loop !12

196:                                              ; preds = %164
  %197 = xor i16 %16, -16191
  store i16 %197, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 482), align 2, !tbaa !10
  %198 = xor i16 %160, -16191
  store i16 %198, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 486), align 2, !tbaa !10
  %199 = xor i16 %63, -16191
  store i16 %199, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 490), align 2, !tbaa !10
  %200 = xor i16 %161, -16191
  store i16 %200, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 494), align 2, !tbaa !10
  %201 = xor i16 %32, -16191
  store i16 %201, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 498), align 2, !tbaa !10
  %202 = xor i16 %162, -16191
  store i16 %202, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 502), align 2, !tbaa !10
  %203 = xor i16 %64, -16191
  store i16 %203, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 506), align 2, !tbaa !10
  %204 = xor i16 %163, -16191
  store i16 %204, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !10
  br label %205

205:                                              ; preds = %196, %66
  %206 = xor i16 %75, %71
  %207 = and i16 %206, 255
  %208 = zext nneg i16 %207 to i64
  %209 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %208
  %210 = load i16, ptr %209, align 2, !tbaa !10
  %211 = lshr i16 %206, 8
  %212 = lshr i16 %210, 8
  %213 = and i16 %210, 255
  %214 = xor i16 %213, %211
  %215 = zext nneg i16 %214 to i64
  %216 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %215
  %217 = load i16, ptr %216, align 2, !tbaa !10
  %218 = xor i16 %217, %212
  %219 = lshr i16 %71, 1
  %220 = and i16 %206, 1
  %221 = icmp eq i16 %220, 0
  %222 = xor i16 %219, -24575
  %223 = select i1 %221, i16 %219, i16 %222
  %224 = lshr i16 %75, 1
  %225 = xor i16 %223, %224
  %226 = lshr i16 %223, 1
  %227 = and i16 %225, 1
  %228 = icmp eq i16 %227, 0
  %229 = xor i16 %226, -24575
  %230 = select i1 %228, i16 %226, i16 %229
  %231 = lshr i16 %75, 2
  %232 = xor i16 %230, %231
  %233 = lshr i16 %230, 1
  %234 = and i16 %232, 1
  %235 = icmp eq i16 %234, 0
  %236 = xor i16 %233, -24575
  %237 = select i1 %235, i16 %233, i16 %236
  %238 = lshr i16 %75, 3
  %239 = xor i16 %237, %238
  %240 = lshr i16 %237, 1
  %241 = and i16 %239, 1
  %242 = icmp eq i16 %241, 0
  %243 = xor i16 %240, -24575
  %244 = select i1 %242, i16 %240, i16 %243
  %245 = lshr i16 %75, 4
  %246 = xor i16 %244, %245
  %247 = lshr i16 %244, 1
  %248 = and i16 %246, 1
  %249 = icmp eq i16 %248, 0
  %250 = xor i16 %247, -24575
  %251 = select i1 %249, i16 %247, i16 %250
  %252 = lshr i16 %75, 5
  %253 = xor i16 %251, %252
  %254 = lshr i16 %251, 1
  %255 = and i16 %253, 1
  %256 = icmp eq i16 %255, 0
  %257 = xor i16 %254, -24575
  %258 = select i1 %256, i16 %254, i16 %257
  %259 = lshr i16 %75, 6
  %260 = xor i16 %258, %259
  %261 = lshr i16 %258, 1
  %262 = and i16 %260, 1
  %263 = icmp eq i16 %262, 0
  %264 = xor i16 %261, -24575
  %265 = select i1 %263, i16 %261, i16 %264
  %266 = lshr i16 %75, 7
  %267 = xor i16 %265, %266
  %268 = lshr i16 %265, 1
  %269 = and i16 %267, 1
  %270 = icmp eq i16 %269, 0
  %271 = xor i16 %268, -24575
  %272 = select i1 %270, i16 %268, i16 %271
  %273 = lshr i16 %75, 8
  %274 = xor i16 %272, %273
  %275 = lshr i16 %272, 1
  %276 = and i16 %274, 1
  %277 = icmp eq i16 %276, 0
  %278 = xor i16 %275, -24575
  %279 = select i1 %277, i16 %275, i16 %278
  %280 = lshr i16 %75, 9
  %281 = xor i16 %279, %280
  %282 = lshr i16 %279, 1
  %283 = and i16 %281, 1
  %284 = icmp eq i16 %283, 0
  %285 = xor i16 %282, -24575
  %286 = select i1 %284, i16 %282, i16 %285
  %287 = lshr i16 %75, 10
  %288 = xor i16 %286, %287
  %289 = lshr i16 %286, 1
  %290 = and i16 %288, 1
  %291 = icmp eq i16 %290, 0
  %292 = xor i16 %289, -24575
  %293 = select i1 %291, i16 %289, i16 %292
  %294 = lshr i16 %75, 11
  %295 = xor i16 %293, %294
  %296 = lshr i16 %293, 1
  %297 = and i16 %295, 1
  %298 = icmp eq i16 %297, 0
  %299 = xor i16 %296, -24575
  %300 = select i1 %298, i16 %296, i16 %299
  %301 = lshr i16 %75, 12
  %302 = xor i16 %300, %301
  %303 = lshr i16 %300, 1
  %304 = and i16 %302, 1
  %305 = icmp eq i16 %304, 0
  %306 = xor i16 %303, -24575
  %307 = select i1 %305, i16 %303, i16 %306
  %308 = lshr i16 %75, 13
  %309 = xor i16 %307, %308
  %310 = lshr i16 %307, 1
  %311 = and i16 %309, 1
  %312 = icmp eq i16 %311, 0
  %313 = xor i16 %310, -24575
  %314 = select i1 %312, i16 %310, i16 %313
  %315 = lshr i16 %75, 14
  %316 = xor i16 %314, %315
  %317 = lshr i16 %314, 1
  %318 = and i16 %316, 1
  %319 = icmp eq i16 %318, 0
  %320 = xor i16 %317, -24575
  %321 = select i1 %319, i16 %317, i16 %320
  %322 = lshr i16 %75, 15
  %323 = lshr i16 %321, 1
  %324 = and i16 %321, 1
  %325 = icmp eq i16 %322, %324
  %326 = xor i16 %323, -24575
  %327 = select i1 %325, i16 %323, i16 %326
  %328 = icmp eq i16 %218, %327
  %329 = select i1 %328, i32 %68, i32 1
  %330 = add nuw nsw i64 %67, 1
  %331 = icmp eq i64 %330, 8
  br i1 %331, label %65, label %66, !llvm.loop !16
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
