; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.le.nodata.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc16.le.nodata.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.crc_initval = internal unnamed_addr constant [8 x i16] [i16 0, i16 1, i16 11, i16 16, i16 129, i16 255, i16 4129, i16 16384], align 2
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
  br label %68

65:                                               ; preds = %282
  %66 = add nuw nsw i64 %69, 1
  %67 = icmp eq i64 %66, 8
  br i1 %67, label %296, label %68, !llvm.loop !6

68:                                               ; preds = %0, %65
  %69 = phi i64 [ 0, %0 ], [ %66, %65 ]
  %70 = getelementptr inbounds nuw i16, ptr @main.crc_initval, i64 %69
  %71 = load i16, ptr %70, align 2, !tbaa !8
  %72 = lshr i16 %71, 1
  %73 = shl nuw nsw i64 1, %69
  %74 = and i64 %73, 137
  %75 = icmp eq i64 %74, 0
  %76 = xor i16 %72, -24575
  %77 = select i1 %75, i16 %76, i16 %72
  %78 = lshr i16 %77, 1
  %79 = and i16 %77, 1
  %80 = icmp eq i16 %79, 0
  %81 = xor i16 %78, -24575
  %82 = select i1 %80, i16 %78, i16 %81
  %83 = lshr i16 %82, 1
  %84 = and i16 %82, 1
  %85 = icmp eq i16 %84, 0
  %86 = xor i16 %83, -24575
  %87 = select i1 %85, i16 %83, i16 %86
  %88 = lshr i16 %87, 1
  %89 = and i16 %87, 1
  %90 = icmp eq i16 %89, 0
  %91 = xor i16 %88, -24575
  %92 = select i1 %90, i16 %88, i16 %91
  %93 = lshr i16 %92, 1
  %94 = and i16 %92, 1
  %95 = icmp eq i16 %94, 0
  %96 = xor i16 %93, -24575
  %97 = select i1 %95, i16 %93, i16 %96
  %98 = lshr i16 %97, 1
  %99 = and i16 %97, 1
  %100 = icmp eq i16 %99, 0
  %101 = xor i16 %98, -24575
  %102 = select i1 %100, i16 %98, i16 %101
  %103 = lshr i16 %102, 1
  %104 = and i16 %102, 1
  %105 = icmp eq i16 %104, 0
  %106 = xor i16 %103, -24575
  %107 = select i1 %105, i16 %103, i16 %106
  %108 = lshr i16 %107, 1
  %109 = and i16 %107, 1
  %110 = icmp eq i16 %109, 0
  %111 = xor i16 %108, -24575
  %112 = select i1 %110, i16 %108, i16 %111
  %113 = lshr i16 %112, 1
  %114 = and i16 %112, 1
  %115 = icmp eq i16 %114, 0
  %116 = xor i16 %113, -24575
  %117 = select i1 %115, i16 %113, i16 %116
  %118 = lshr i16 %117, 1
  %119 = and i16 %117, 1
  %120 = icmp eq i16 %119, 0
  %121 = xor i16 %118, -24575
  %122 = select i1 %120, i16 %118, i16 %121
  %123 = lshr i16 %122, 1
  %124 = and i16 %122, 1
  %125 = icmp eq i16 %124, 0
  %126 = xor i16 %123, -24575
  %127 = select i1 %125, i16 %123, i16 %126
  %128 = lshr i16 %127, 1
  %129 = and i16 %127, 1
  %130 = icmp eq i16 %129, 0
  %131 = xor i16 %128, -24575
  %132 = select i1 %130, i16 %128, i16 %131
  %133 = lshr i16 %132, 1
  %134 = and i16 %132, 1
  %135 = icmp eq i16 %134, 0
  %136 = xor i16 %133, -24575
  %137 = select i1 %135, i16 %133, i16 %136
  %138 = lshr i16 %137, 1
  %139 = and i16 %137, 1
  %140 = icmp eq i16 %139, 0
  %141 = xor i16 %138, -24575
  %142 = select i1 %140, i16 %138, i16 %141
  %143 = lshr i16 %142, 1
  %144 = and i16 %142, 1
  %145 = icmp eq i16 %144, 0
  %146 = xor i16 %143, -24575
  %147 = select i1 %145, i16 %143, i16 %146
  %148 = lshr i16 %147, 1
  %149 = and i16 %147, 1
  %150 = icmp eq i16 %149, 0
  %151 = xor i16 %148, -24575
  %152 = select i1 %150, i16 %148, i16 %151
  %153 = load i16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !8
  %154 = icmp eq i16 %153, 0
  br i1 %154, label %155, label %282

155:                                              ; preds = %68
  store i16 %2, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 64, !tbaa !8
  store i16 %3, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 64, !tbaa !8
  store i16 %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 64, !tbaa !8
  store i16 %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 64, !tbaa !8
  store i16 %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 64, !tbaa !8
  store i16 %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 64, !tbaa !8
  store i16 %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 448), align 64, !tbaa !8
  store i16 %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 32, !tbaa !8
  store i16 %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 32, !tbaa !8
  store i16 %11, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 32, !tbaa !8
  store i16 %12, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 32, !tbaa !8
  store i16 %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 288), align 32, !tbaa !8
  store i16 %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 352), align 32, !tbaa !8
  store i16 %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 416), align 32, !tbaa !8
  store i16 %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 480), align 32, !tbaa !8
  store i16 %17, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !8
  store i16 %18, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !8
  store i16 %19, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !8
  store i16 %20, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !8
  store i16 %21, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !8
  store i16 %22, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !8
  store i16 %23, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !8
  store i16 %24, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !8
  store i16 %25, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 272), align 16, !tbaa !8
  store i16 %26, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 304), align 16, !tbaa !8
  store i16 %27, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 336), align 16, !tbaa !8
  store i16 %28, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 368), align 16, !tbaa !8
  store i16 %29, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 400), align 16, !tbaa !8
  store i16 %30, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 432), align 16, !tbaa !8
  store i16 %31, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 464), align 16, !tbaa !8
  store i16 %32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 496), align 16, !tbaa !8
  store i16 %33, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 8), align 8, !tbaa !8
  store i16 %34, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 24), align 8, !tbaa !8
  store i16 %35, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 40), align 8, !tbaa !8
  store i16 %36, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 56), align 8, !tbaa !8
  store i16 %37, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 72), align 8, !tbaa !8
  store i16 %38, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 88), align 8, !tbaa !8
  store i16 %39, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 104), align 8, !tbaa !8
  store i16 %40, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 120), align 8, !tbaa !8
  store i16 %41, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 136), align 8, !tbaa !8
  store i16 %42, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 152), align 8, !tbaa !8
  store i16 %43, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 168), align 8, !tbaa !8
  store i16 %44, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 184), align 8, !tbaa !8
  store i16 %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 200), align 8, !tbaa !8
  store i16 %46, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 216), align 8, !tbaa !8
  store i16 %47, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 232), align 8, !tbaa !8
  store i16 %48, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 248), align 8, !tbaa !8
  store i16 %49, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 264), align 8, !tbaa !8
  store i16 %50, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 280), align 8, !tbaa !8
  store i16 %51, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 296), align 8, !tbaa !8
  store i16 %52, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 312), align 8, !tbaa !8
  store i16 %53, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 328), align 8, !tbaa !8
  store i16 %54, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 344), align 8, !tbaa !8
  store i16 %55, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 360), align 8, !tbaa !8
  store i16 %56, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 376), align 8, !tbaa !8
  store i16 %57, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 392), align 8, !tbaa !8
  store i16 %58, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 408), align 8, !tbaa !8
  store i16 %59, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 424), align 8, !tbaa !8
  store i16 %60, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 440), align 8, !tbaa !8
  store i16 %61, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 456), align 8, !tbaa !8
  store i16 %62, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 472), align 8, !tbaa !8
  store i16 %63, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 488), align 8, !tbaa !8
  store i16 %64, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 504), align 8, !tbaa !8
  %156 = load <29 x i16>, ptr @CRCTable, align 64, !tbaa !8
  %157 = shufflevector <29 x i16> %156, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %158 = xor <8 x i16> %157, splat (i16 -15999)
  %159 = extractelement <8 x i16> %158, i64 0
  store i16 %159, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 4), align 4, !tbaa !8
  %160 = extractelement <8 x i16> %158, i64 1
  store i16 %160, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 12), align 4, !tbaa !8
  %161 = extractelement <8 x i16> %158, i64 2
  store i16 %161, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 20), align 4, !tbaa !8
  %162 = extractelement <8 x i16> %158, i64 3
  store i16 %162, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 28), align 4, !tbaa !8
  %163 = extractelement <8 x i16> %158, i64 4
  store i16 %163, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 36), align 4, !tbaa !8
  %164 = extractelement <8 x i16> %158, i64 5
  store i16 %164, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 44), align 4, !tbaa !8
  %165 = extractelement <8 x i16> %158, i64 6
  store i16 %165, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 52), align 4, !tbaa !8
  %166 = extractelement <8 x i16> %158, i64 7
  store i16 %166, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 60), align 4, !tbaa !8
  %167 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 64, !tbaa !8
  %168 = shufflevector <29 x i16> %167, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %169 = xor <8 x i16> %168, splat (i16 -15999)
  %170 = extractelement <8 x i16> %169, i64 0
  store i16 %170, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 68), align 4, !tbaa !8
  %171 = extractelement <8 x i16> %169, i64 1
  store i16 %171, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 76), align 4, !tbaa !8
  %172 = extractelement <8 x i16> %169, i64 2
  store i16 %172, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 84), align 4, !tbaa !8
  %173 = extractelement <8 x i16> %169, i64 3
  store i16 %173, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 92), align 4, !tbaa !8
  %174 = extractelement <8 x i16> %169, i64 4
  store i16 %174, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 100), align 4, !tbaa !8
  %175 = extractelement <8 x i16> %169, i64 5
  store i16 %175, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 108), align 4, !tbaa !8
  %176 = extractelement <8 x i16> %169, i64 6
  store i16 %176, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 116), align 4, !tbaa !8
  %177 = extractelement <8 x i16> %169, i64 7
  store i16 %177, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 124), align 4, !tbaa !8
  %178 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 64, !tbaa !8
  %179 = shufflevector <29 x i16> %178, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %180 = xor <8 x i16> %179, splat (i16 -15999)
  %181 = extractelement <8 x i16> %180, i64 0
  store i16 %181, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 132), align 4, !tbaa !8
  %182 = extractelement <8 x i16> %180, i64 1
  store i16 %182, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 140), align 4, !tbaa !8
  %183 = extractelement <8 x i16> %180, i64 2
  store i16 %183, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 148), align 4, !tbaa !8
  %184 = extractelement <8 x i16> %180, i64 3
  store i16 %184, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 156), align 4, !tbaa !8
  %185 = extractelement <8 x i16> %180, i64 4
  store i16 %185, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 164), align 4, !tbaa !8
  %186 = extractelement <8 x i16> %180, i64 5
  store i16 %186, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 172), align 4, !tbaa !8
  %187 = extractelement <8 x i16> %180, i64 6
  store i16 %187, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 180), align 4, !tbaa !8
  %188 = extractelement <8 x i16> %180, i64 7
  store i16 %188, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 188), align 4, !tbaa !8
  %189 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 64, !tbaa !8
  %190 = shufflevector <29 x i16> %189, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %191 = xor <8 x i16> %190, splat (i16 -15999)
  %192 = extractelement <8 x i16> %191, i64 0
  store i16 %192, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 196), align 4, !tbaa !8
  %193 = extractelement <8 x i16> %191, i64 1
  store i16 %193, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 204), align 4, !tbaa !8
  %194 = extractelement <8 x i16> %191, i64 2
  store i16 %194, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 212), align 4, !tbaa !8
  %195 = extractelement <8 x i16> %191, i64 3
  store i16 %195, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 220), align 4, !tbaa !8
  %196 = extractelement <8 x i16> %191, i64 4
  store i16 %196, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 228), align 4, !tbaa !8
  %197 = extractelement <8 x i16> %191, i64 5
  store i16 %197, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 236), align 4, !tbaa !8
  %198 = extractelement <8 x i16> %191, i64 6
  store i16 %198, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 244), align 4, !tbaa !8
  %199 = extractelement <8 x i16> %191, i64 7
  store i16 %199, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 252), align 4, !tbaa !8
  %200 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 64, !tbaa !8
  %201 = shufflevector <29 x i16> %200, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %202 = xor <8 x i16> %201, splat (i16 -15999)
  %203 = extractelement <8 x i16> %202, i64 0
  store i16 %203, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 260), align 4, !tbaa !8
  %204 = extractelement <8 x i16> %202, i64 1
  store i16 %204, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 268), align 4, !tbaa !8
  %205 = extractelement <8 x i16> %202, i64 2
  store i16 %205, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 276), align 4, !tbaa !8
  %206 = extractelement <8 x i16> %202, i64 3
  store i16 %206, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 284), align 4, !tbaa !8
  %207 = extractelement <8 x i16> %202, i64 4
  store i16 %207, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 292), align 4, !tbaa !8
  %208 = extractelement <8 x i16> %202, i64 5
  store i16 %208, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 300), align 4, !tbaa !8
  %209 = extractelement <8 x i16> %202, i64 6
  store i16 %209, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 308), align 4, !tbaa !8
  %210 = extractelement <8 x i16> %202, i64 7
  store i16 %210, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 316), align 4, !tbaa !8
  %211 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 64, !tbaa !8
  %212 = shufflevector <29 x i16> %211, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %213 = xor <8 x i16> %212, splat (i16 -15999)
  %214 = extractelement <8 x i16> %213, i64 0
  store i16 %214, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 324), align 4, !tbaa !8
  %215 = extractelement <8 x i16> %213, i64 1
  store i16 %215, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 332), align 4, !tbaa !8
  %216 = extractelement <8 x i16> %213, i64 2
  store i16 %216, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 340), align 4, !tbaa !8
  %217 = extractelement <8 x i16> %213, i64 3
  store i16 %217, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 348), align 4, !tbaa !8
  %218 = extractelement <8 x i16> %213, i64 4
  store i16 %218, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 356), align 4, !tbaa !8
  %219 = extractelement <8 x i16> %213, i64 5
  store i16 %219, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 364), align 4, !tbaa !8
  %220 = extractelement <8 x i16> %213, i64 6
  store i16 %220, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 372), align 4, !tbaa !8
  %221 = extractelement <8 x i16> %213, i64 7
  store i16 %221, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 380), align 4, !tbaa !8
  %222 = load <29 x i16>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 64, !tbaa !8
  %223 = shufflevector <29 x i16> %222, <29 x i16> poison, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %224 = xor <8 x i16> %223, splat (i16 -15999)
  %225 = extractelement <8 x i16> %224, i64 0
  store i16 %225, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 388), align 4, !tbaa !8
  %226 = extractelement <8 x i16> %224, i64 1
  store i16 %226, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 396), align 4, !tbaa !8
  %227 = extractelement <8 x i16> %224, i64 2
  store i16 %227, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 404), align 4, !tbaa !8
  %228 = extractelement <8 x i16> %224, i64 3
  store i16 %228, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 412), align 4, !tbaa !8
  %229 = extractelement <8 x i16> %224, i64 4
  store i16 %229, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 420), align 4, !tbaa !8
  %230 = extractelement <8 x i16> %224, i64 5
  store i16 %230, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 428), align 4, !tbaa !8
  %231 = extractelement <8 x i16> %224, i64 6
  store i16 %231, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 436), align 4, !tbaa !8
  %232 = extractelement <8 x i16> %224, i64 7
  store i16 %232, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 444), align 4, !tbaa !8
  %233 = xor i16 %8, -15999
  store i16 %233, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 452), align 4, !tbaa !8
  %234 = xor i16 %61, -15999
  store i16 %234, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 460), align 4, !tbaa !8
  %235 = xor i16 %31, -15999
  store i16 %235, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 468), align 4, !tbaa !8
  %236 = xor i16 %62, -15999
  store i16 %236, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 476), align 4, !tbaa !8
  %237 = xor i16 %16, -15999
  store i16 %237, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 484), align 4, !tbaa !8
  %238 = xor i16 %63, -15999
  store i16 %238, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 492), align 4, !tbaa !8
  %239 = xor i16 %32, -15999
  store i16 %239, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 500), align 4, !tbaa !8
  %240 = xor i16 %64, -15999
  store i16 %240, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 508), align 4, !tbaa !8
  br label %241

241:                                              ; preds = %155, %241
  %242 = phi i64 [ %271, %241 ], [ 0, %155 ]
  %243 = shl i64 %242, 1
  %244 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %243
  %245 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %243
  %246 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %243
  %247 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %243
  %248 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %243
  %249 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %243
  %250 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %243
  %251 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %243
  %252 = load <15 x i16>, ptr %244, align 32, !tbaa !8
  %253 = shufflevector <15 x i16> %252, <15 x i16> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %254 = xor <8 x i16> %253, splat (i16 -16191)
  %255 = getelementptr inbounds nuw i8, ptr %244, i64 2
  %256 = getelementptr inbounds nuw i8, ptr %245, i64 6
  %257 = getelementptr inbounds nuw i8, ptr %246, i64 10
  %258 = getelementptr inbounds nuw i8, ptr %247, i64 14
  %259 = getelementptr inbounds nuw i8, ptr %248, i64 18
  %260 = getelementptr inbounds nuw i8, ptr %249, i64 22
  %261 = getelementptr inbounds nuw i8, ptr %250, i64 26
  %262 = getelementptr inbounds nuw i8, ptr %251, i64 30
  %263 = extractelement <8 x i16> %254, i64 0
  store i16 %263, ptr %255, align 2, !tbaa !8
  %264 = extractelement <8 x i16> %254, i64 1
  store i16 %264, ptr %256, align 2, !tbaa !8
  %265 = extractelement <8 x i16> %254, i64 2
  store i16 %265, ptr %257, align 2, !tbaa !8
  %266 = extractelement <8 x i16> %254, i64 3
  store i16 %266, ptr %258, align 2, !tbaa !8
  %267 = extractelement <8 x i16> %254, i64 4
  store i16 %267, ptr %259, align 2, !tbaa !8
  %268 = extractelement <8 x i16> %254, i64 5
  store i16 %268, ptr %260, align 2, !tbaa !8
  %269 = extractelement <8 x i16> %254, i64 6
  store i16 %269, ptr %261, align 2, !tbaa !8
  %270 = extractelement <8 x i16> %254, i64 7
  store i16 %270, ptr %262, align 2, !tbaa !8
  %271 = add nuw i64 %242, 8
  %272 = icmp eq i64 %271, 120
  br i1 %272, label %273, label %241, !llvm.loop !12

273:                                              ; preds = %241
  %274 = xor i16 %16, -16191
  store i16 %274, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 482), align 2, !tbaa !8
  %275 = xor i16 %237, -16191
  store i16 %275, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 486), align 2, !tbaa !8
  %276 = xor i16 %63, -16191
  store i16 %276, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 490), align 2, !tbaa !8
  %277 = xor i16 %238, -16191
  store i16 %277, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 494), align 2, !tbaa !8
  %278 = xor i16 %32, -16191
  store i16 %278, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 498), align 2, !tbaa !8
  %279 = xor i16 %239, -16191
  store i16 %279, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 502), align 2, !tbaa !8
  %280 = xor i16 %64, -16191
  store i16 %280, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 506), align 2, !tbaa !8
  %281 = xor i16 %240, -16191
  store i16 %281, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 510), align 2, !tbaa !8
  br label %282

282:                                              ; preds = %273, %68
  %283 = lshr i16 %71, 8
  %284 = and i16 %71, 255
  %285 = zext nneg i16 %284 to i64
  %286 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %285
  %287 = load i16, ptr %286, align 2, !tbaa !8
  %288 = lshr i16 %287, 8
  %289 = and i16 %287, 255
  %290 = xor i16 %289, %283
  %291 = zext nneg i16 %290 to i64
  %292 = getelementptr inbounds nuw i16, ptr @CRCTable, i64 %291
  %293 = load i16, ptr %292, align 2, !tbaa !8
  %294 = xor i16 %293, %288
  %295 = icmp eq i16 %152, %294
  br i1 %295, label %65, label %296

296:                                              ; preds = %65, %282
  %297 = phi i32 [ 1, %282 ], [ 0, %65 ]
  ret i32 %297
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
!12 = distinct !{!12, !7, !13, !14}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
