; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc32.be.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/HashRecognize/crc32.be.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@main.sample = internal unnamed_addr constant [8 x i32] [i32 0, i32 1, i32 11, i32 16, i32 129, i32 142, i32 196, i32 255], align 4
@CRCTable = internal unnamed_addr global [256 x i32] zeroinitializer, align 16

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @CRCTable, align 16
  %2 = insertelement <4 x i32> poison, i32 %1, i64 0
  %3 = shufflevector <4 x i32> %2, <4 x i32> poison, <4 x i32> zeroinitializer
  %4 = xor <4 x i32> %3, <i32 33800, i32 67600, i32 101400, i32 135200>
  %5 = xor <4 x i32> %3, <i32 169000, i32 202800, i32 236600, i32 270400>
  %6 = xor <4 x i32> %3, <i32 304200, i32 338000, i32 371800, i32 405600>
  %7 = xor <4 x i32> %3, <i32 439400, i32 473200, i32 507000, i32 540800>
  %8 = xor <4 x i32> %3, <i32 574600, i32 608400, i32 642200, i32 676000>
  %9 = xor <4 x i32> %3, <i32 709800, i32 743600, i32 777400, i32 811200>
  %10 = xor <4 x i32> %3, <i32 845000, i32 878800, i32 912600, i32 946400>
  %11 = xor <4 x i32> %3, <i32 980200, i32 1014000, i32 1047800, i32 1081600>
  %12 = xor <4 x i32> %3, <i32 1049864, i32 1149200, i32 1117464, i32 1216800>
  %13 = xor <4 x i32> %3, <i32 1185064, i32 1284400, i32 1252664, i32 1352000>
  %14 = xor <4 x i32> %3, <i32 1320264, i32 1419600, i32 1387864, i32 1487200>
  %15 = xor <4 x i32> %3, <i32 1455464, i32 1554800, i32 1523064, i32 1622400>
  %16 = xor <4 x i32> %3, <i32 1590664, i32 1690000, i32 1658264, i32 1757600>
  %17 = xor <4 x i32> %3, <i32 1725864, i32 1825200, i32 1793464, i32 1892800>
  %18 = xor <4 x i32> %3, <i32 1861064, i32 1960400, i32 1928664, i32 2028000>
  %19 = insertelement <2 x i32> poison, i32 %1, i64 0
  %20 = shufflevector <2 x i32> %19, <2 x i32> poison, <2 x i32> zeroinitializer
  %21 = xor <2 x i32> %20, <i32 1996264, i32 2095600>
  %22 = xor i32 %1, 2063864
  br label %24

23:                                               ; preds = %115
  ret i32 %336

24:                                               ; preds = %0, %115
  %25 = phi i64 [ 0, %0 ], [ %337, %115 ]
  %26 = phi i32 [ 0, %0 ], [ %336, %115 ]
  %27 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %25
  %28 = load i32, ptr %27, align 4, !tbaa !6
  %29 = sub nuw nsw i64 7, %25
  %30 = getelementptr inbounds nuw i32, ptr @main.sample, i64 %29
  %31 = load i32, ptr %30, align 4, !tbaa !6
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 1020), align 4, !tbaa !6
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %34, label %115

34:                                               ; preds = %24
  store <4 x i32> %4, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 4), align 4, !tbaa !6
  store <4 x i32> %5, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 20), align 4, !tbaa !6
  store <4 x i32> %6, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 36), align 4, !tbaa !6
  store <4 x i32> %7, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 52), align 4, !tbaa !6
  store <4 x i32> %8, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 68), align 4, !tbaa !6
  store <4 x i32> %9, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 84), align 4, !tbaa !6
  store <4 x i32> %10, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 100), align 4, !tbaa !6
  store <4 x i32> %11, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 116), align 4, !tbaa !6
  store <4 x i32> %12, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 132), align 4, !tbaa !6
  store <4 x i32> %13, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 148), align 4, !tbaa !6
  store <4 x i32> %14, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 164), align 4, !tbaa !6
  store <4 x i32> %15, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 180), align 4, !tbaa !6
  store <4 x i32> %16, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 196), align 4, !tbaa !6
  store <4 x i32> %17, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 212), align 4, !tbaa !6
  store <4 x i32> %18, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 228), align 4, !tbaa !6
  store <2 x i32> %21, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 244), align 4, !tbaa !6
  store i32 %22, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 252), align 4, !tbaa !6
  %35 = load <4 x i32>, ptr @CRCTable, align 16, !tbaa !6
  %36 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !6
  %37 = xor <4 x i32> %35, splat (i32 2163200)
  %38 = xor <4 x i32> %36, splat (i32 2163200)
  store <4 x i32> %37, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 256), align 16, !tbaa !6
  store <4 x i32> %38, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 272), align 16, !tbaa !6
  %39 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !6
  %40 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !6
  %41 = xor <4 x i32> %39, splat (i32 2163200)
  %42 = xor <4 x i32> %40, splat (i32 2163200)
  store <4 x i32> %41, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 288), align 16, !tbaa !6
  store <4 x i32> %42, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 304), align 16, !tbaa !6
  %43 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !6
  %44 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !6
  %45 = xor <4 x i32> %43, splat (i32 2163200)
  %46 = xor <4 x i32> %44, splat (i32 2163200)
  store <4 x i32> %45, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 320), align 16, !tbaa !6
  store <4 x i32> %46, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 336), align 16, !tbaa !6
  %47 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !6
  %48 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !6
  %49 = xor <4 x i32> %47, splat (i32 2163200)
  %50 = xor <4 x i32> %48, splat (i32 2163200)
  store <4 x i32> %49, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 352), align 16, !tbaa !6
  store <4 x i32> %50, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 368), align 16, !tbaa !6
  %51 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 16, !tbaa !6
  %52 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !6
  %53 = xor <4 x i32> %51, splat (i32 2163200)
  %54 = xor <4 x i32> %52, splat (i32 2163200)
  store <4 x i32> %53, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 384), align 16, !tbaa !6
  store <4 x i32> %54, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 400), align 16, !tbaa !6
  %55 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 16, !tbaa !6
  %56 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !6
  %57 = xor <4 x i32> %55, splat (i32 2163200)
  %58 = xor <4 x i32> %56, splat (i32 2163200)
  store <4 x i32> %57, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 416), align 16, !tbaa !6
  store <4 x i32> %58, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 432), align 16, !tbaa !6
  %59 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 16, !tbaa !6
  %60 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !6
  %61 = xor <4 x i32> %59, splat (i32 2163200)
  %62 = xor <4 x i32> %60, splat (i32 2163200)
  store <4 x i32> %61, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 448), align 16, !tbaa !6
  store <4 x i32> %62, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 464), align 16, !tbaa !6
  %63 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 16, !tbaa !6
  %64 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !6
  %65 = xor <4 x i32> %63, splat (i32 2163200)
  %66 = xor <4 x i32> %64, splat (i32 2163200)
  store <4 x i32> %65, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 480), align 16, !tbaa !6
  store <4 x i32> %66, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 496), align 16, !tbaa !6
  %67 = load <4 x i32>, ptr @CRCTable, align 16, !tbaa !6
  %68 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 16), align 16, !tbaa !6
  %69 = xor <4 x i32> %67, splat (i32 4326400)
  %70 = xor <4 x i32> %68, splat (i32 4326400)
  store <4 x i32> %69, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 512), align 16, !tbaa !6
  store <4 x i32> %70, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 528), align 16, !tbaa !6
  %71 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 32), align 16, !tbaa !6
  %72 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 48), align 16, !tbaa !6
  %73 = xor <4 x i32> %71, splat (i32 4326400)
  %74 = xor <4 x i32> %72, splat (i32 4326400)
  store <4 x i32> %73, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 544), align 16, !tbaa !6
  store <4 x i32> %74, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 560), align 16, !tbaa !6
  %75 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 64), align 16, !tbaa !6
  %76 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 80), align 16, !tbaa !6
  %77 = xor <4 x i32> %75, splat (i32 4326400)
  %78 = xor <4 x i32> %76, splat (i32 4326400)
  store <4 x i32> %77, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 576), align 16, !tbaa !6
  store <4 x i32> %78, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 592), align 16, !tbaa !6
  %79 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 96), align 16, !tbaa !6
  %80 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 112), align 16, !tbaa !6
  %81 = xor <4 x i32> %79, splat (i32 4326400)
  %82 = xor <4 x i32> %80, splat (i32 4326400)
  store <4 x i32> %81, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 608), align 16, !tbaa !6
  store <4 x i32> %82, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 624), align 16, !tbaa !6
  %83 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 128), align 16, !tbaa !6
  %84 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 144), align 16, !tbaa !6
  %85 = xor <4 x i32> %83, splat (i32 4326400)
  %86 = xor <4 x i32> %84, splat (i32 4326400)
  store <4 x i32> %85, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 640), align 16, !tbaa !6
  store <4 x i32> %86, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 656), align 16, !tbaa !6
  %87 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 160), align 16, !tbaa !6
  %88 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 176), align 16, !tbaa !6
  %89 = xor <4 x i32> %87, splat (i32 4326400)
  %90 = xor <4 x i32> %88, splat (i32 4326400)
  store <4 x i32> %89, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 672), align 16, !tbaa !6
  store <4 x i32> %90, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 688), align 16, !tbaa !6
  %91 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 192), align 16, !tbaa !6
  %92 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 208), align 16, !tbaa !6
  %93 = xor <4 x i32> %91, splat (i32 4326400)
  %94 = xor <4 x i32> %92, splat (i32 4326400)
  store <4 x i32> %93, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 704), align 16, !tbaa !6
  store <4 x i32> %94, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 720), align 16, !tbaa !6
  %95 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 224), align 16, !tbaa !6
  %96 = load <4 x i32>, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 240), align 16, !tbaa !6
  %97 = xor <4 x i32> %95, splat (i32 4326400)
  %98 = xor <4 x i32> %96, splat (i32 4326400)
  store <4 x i32> %97, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 736), align 16, !tbaa !6
  store <4 x i32> %98, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 752), align 16, !tbaa !6
  %99 = xor <4 x i32> %37, splat (i32 4326400)
  %100 = xor <4 x i32> %38, splat (i32 4326400)
  store <4 x i32> %99, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 768), align 16, !tbaa !6
  store <4 x i32> %100, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 784), align 16, !tbaa !6
  %101 = xor <4 x i32> %41, splat (i32 4326400)
  %102 = xor <4 x i32> %42, splat (i32 4326400)
  store <4 x i32> %101, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 800), align 16, !tbaa !6
  store <4 x i32> %102, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 816), align 16, !tbaa !6
  %103 = xor <4 x i32> %45, splat (i32 4326400)
  %104 = xor <4 x i32> %46, splat (i32 4326400)
  store <4 x i32> %103, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 832), align 16, !tbaa !6
  store <4 x i32> %104, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 848), align 16, !tbaa !6
  %105 = xor <4 x i32> %49, splat (i32 4326400)
  %106 = xor <4 x i32> %50, splat (i32 4326400)
  store <4 x i32> %105, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 864), align 16, !tbaa !6
  store <4 x i32> %106, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 880), align 16, !tbaa !6
  %107 = xor <4 x i32> %53, splat (i32 4326400)
  %108 = xor <4 x i32> %54, splat (i32 4326400)
  store <4 x i32> %107, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 896), align 16, !tbaa !6
  store <4 x i32> %108, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 912), align 16, !tbaa !6
  %109 = xor <4 x i32> %57, splat (i32 4326400)
  %110 = xor <4 x i32> %58, splat (i32 4326400)
  store <4 x i32> %109, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 928), align 16, !tbaa !6
  store <4 x i32> %110, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 944), align 16, !tbaa !6
  %111 = xor <4 x i32> %61, splat (i32 4326400)
  %112 = xor <4 x i32> %62, splat (i32 4326400)
  store <4 x i32> %111, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 960), align 16, !tbaa !6
  store <4 x i32> %112, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 976), align 16, !tbaa !6
  %113 = xor <4 x i32> %65, splat (i32 4326400)
  %114 = xor <4 x i32> %66, splat (i32 4326400)
  store <4 x i32> %113, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 992), align 16, !tbaa !6
  store <4 x i32> %114, ptr getelementptr inbounds nuw (i8, ptr @CRCTable, i64 1008), align 16, !tbaa !6
  br label %115

115:                                              ; preds = %34, %24
  %116 = xor i32 %31, %28
  %117 = lshr i32 %116, 24
  %118 = zext nneg i32 %117 to i64
  %119 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %118
  %120 = load i32, ptr %119, align 4, !tbaa !6
  %121 = shl i32 %116, 8
  %122 = xor i32 %120, %121
  %123 = lshr i32 %122, 24
  %124 = zext nneg i32 %123 to i64
  %125 = shl i32 %28, 16
  %126 = shl i32 %120, 8
  %127 = xor i32 %126, %125
  %128 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %124
  %129 = load i32, ptr %128, align 4, !tbaa !6
  %130 = xor i32 %127, %129
  %131 = shl i32 %31, 16
  %132 = xor i32 %130, %131
  %133 = lshr i32 %132, 24
  %134 = zext nneg i32 %133 to i64
  %135 = shl i32 %130, 8
  %136 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %134
  %137 = load i32, ptr %136, align 4, !tbaa !6
  %138 = xor i32 %135, %137
  %139 = shl i32 %31, 24
  %140 = xor i32 %138, %139
  %141 = lshr i32 %140, 24
  %142 = zext nneg i32 %141 to i64
  %143 = shl i32 %138, 8
  %144 = getelementptr inbounds nuw i32, ptr @CRCTable, i64 %142
  %145 = load i32, ptr %144, align 4, !tbaa !6
  %146 = xor i32 %143, %145
  %147 = shl i32 %28, 1
  %148 = xor i32 %147, 33800
  %149 = icmp slt i32 %116, 0
  %150 = select i1 %149, i32 %148, i32 %147
  %151 = shl i32 %31, 1
  %152 = xor i32 %150, %151
  %153 = shl i32 %150, 1
  %154 = xor i32 %153, 33800
  %155 = icmp slt i32 %152, 0
  %156 = select i1 %155, i32 %154, i32 %153
  %157 = shl i32 %31, 2
  %158 = xor i32 %156, %157
  %159 = shl i32 %156, 1
  %160 = xor i32 %159, 33800
  %161 = icmp slt i32 %158, 0
  %162 = select i1 %161, i32 %160, i32 %159
  %163 = shl i32 %31, 3
  %164 = xor i32 %162, %163
  %165 = shl i32 %162, 1
  %166 = xor i32 %165, 33800
  %167 = icmp slt i32 %164, 0
  %168 = select i1 %167, i32 %166, i32 %165
  %169 = shl i32 %31, 4
  %170 = xor i32 %168, %169
  %171 = shl i32 %168, 1
  %172 = xor i32 %171, 33800
  %173 = icmp slt i32 %170, 0
  %174 = select i1 %173, i32 %172, i32 %171
  %175 = shl i32 %31, 5
  %176 = xor i32 %174, %175
  %177 = shl i32 %174, 1
  %178 = xor i32 %177, 33800
  %179 = icmp slt i32 %176, 0
  %180 = select i1 %179, i32 %178, i32 %177
  %181 = shl i32 %31, 6
  %182 = xor i32 %180, %181
  %183 = shl i32 %180, 1
  %184 = xor i32 %183, 33800
  %185 = icmp slt i32 %182, 0
  %186 = select i1 %185, i32 %184, i32 %183
  %187 = shl i32 %31, 7
  %188 = xor i32 %186, %187
  %189 = shl i32 %186, 1
  %190 = xor i32 %189, 33800
  %191 = icmp slt i32 %188, 0
  %192 = select i1 %191, i32 %190, i32 %189
  %193 = shl i32 %31, 8
  %194 = xor i32 %192, %193
  %195 = shl i32 %192, 1
  %196 = xor i32 %195, 33800
  %197 = icmp slt i32 %194, 0
  %198 = select i1 %197, i32 %196, i32 %195
  %199 = shl i32 %31, 9
  %200 = xor i32 %198, %199
  %201 = shl i32 %198, 1
  %202 = xor i32 %201, 33800
  %203 = icmp slt i32 %200, 0
  %204 = select i1 %203, i32 %202, i32 %201
  %205 = shl i32 %31, 10
  %206 = xor i32 %204, %205
  %207 = shl i32 %204, 1
  %208 = xor i32 %207, 33800
  %209 = icmp slt i32 %206, 0
  %210 = select i1 %209, i32 %208, i32 %207
  %211 = shl i32 %31, 11
  %212 = xor i32 %210, %211
  %213 = shl i32 %210, 1
  %214 = xor i32 %213, 33800
  %215 = icmp slt i32 %212, 0
  %216 = select i1 %215, i32 %214, i32 %213
  %217 = shl i32 %31, 12
  %218 = xor i32 %216, %217
  %219 = shl i32 %216, 1
  %220 = xor i32 %219, 33800
  %221 = icmp slt i32 %218, 0
  %222 = select i1 %221, i32 %220, i32 %219
  %223 = shl i32 %31, 13
  %224 = xor i32 %222, %223
  %225 = shl i32 %222, 1
  %226 = xor i32 %225, 33800
  %227 = icmp slt i32 %224, 0
  %228 = select i1 %227, i32 %226, i32 %225
  %229 = shl i32 %31, 14
  %230 = xor i32 %228, %229
  %231 = shl i32 %228, 1
  %232 = xor i32 %231, 33800
  %233 = icmp slt i32 %230, 0
  %234 = select i1 %233, i32 %232, i32 %231
  %235 = shl i32 %31, 15
  %236 = xor i32 %234, %235
  %237 = shl i32 %234, 1
  %238 = xor i32 %237, 33800
  %239 = icmp slt i32 %236, 0
  %240 = select i1 %239, i32 %238, i32 %237
  %241 = xor i32 %240, %131
  %242 = shl i32 %240, 1
  %243 = xor i32 %242, 33800
  %244 = icmp slt i32 %241, 0
  %245 = select i1 %244, i32 %243, i32 %242
  %246 = shl i32 %31, 17
  %247 = xor i32 %245, %246
  %248 = shl i32 %245, 1
  %249 = xor i32 %248, 33800
  %250 = icmp slt i32 %247, 0
  %251 = select i1 %250, i32 %249, i32 %248
  %252 = shl i32 %31, 18
  %253 = xor i32 %251, %252
  %254 = shl i32 %251, 1
  %255 = xor i32 %254, 33800
  %256 = icmp slt i32 %253, 0
  %257 = select i1 %256, i32 %255, i32 %254
  %258 = shl i32 %31, 19
  %259 = xor i32 %257, %258
  %260 = shl i32 %257, 1
  %261 = xor i32 %260, 33800
  %262 = icmp slt i32 %259, 0
  %263 = select i1 %262, i32 %261, i32 %260
  %264 = shl i32 %31, 20
  %265 = xor i32 %263, %264
  %266 = shl i32 %263, 1
  %267 = xor i32 %266, 33800
  %268 = icmp slt i32 %265, 0
  %269 = select i1 %268, i32 %267, i32 %266
  %270 = shl i32 %31, 21
  %271 = xor i32 %269, %270
  %272 = shl i32 %269, 1
  %273 = xor i32 %272, 33800
  %274 = icmp slt i32 %271, 0
  %275 = select i1 %274, i32 %273, i32 %272
  %276 = shl i32 %31, 22
  %277 = xor i32 %275, %276
  %278 = shl i32 %275, 1
  %279 = xor i32 %278, 33800
  %280 = icmp slt i32 %277, 0
  %281 = select i1 %280, i32 %279, i32 %278
  %282 = shl i32 %31, 23
  %283 = xor i32 %281, %282
  %284 = shl i32 %281, 1
  %285 = xor i32 %284, 33800
  %286 = icmp slt i32 %283, 0
  %287 = select i1 %286, i32 %285, i32 %284
  %288 = xor i32 %287, %139
  %289 = shl i32 %287, 1
  %290 = xor i32 %289, 33800
  %291 = icmp slt i32 %288, 0
  %292 = select i1 %291, i32 %290, i32 %289
  %293 = shl i32 %31, 25
  %294 = xor i32 %292, %293
  %295 = shl i32 %292, 1
  %296 = xor i32 %295, 33800
  %297 = icmp slt i32 %294, 0
  %298 = select i1 %297, i32 %296, i32 %295
  %299 = shl i32 %31, 26
  %300 = xor i32 %298, %299
  %301 = shl i32 %298, 1
  %302 = xor i32 %301, 33800
  %303 = icmp slt i32 %300, 0
  %304 = select i1 %303, i32 %302, i32 %301
  %305 = shl i32 %31, 27
  %306 = xor i32 %304, %305
  %307 = shl i32 %304, 1
  %308 = xor i32 %307, 33800
  %309 = icmp slt i32 %306, 0
  %310 = select i1 %309, i32 %308, i32 %307
  %311 = shl i32 %31, 28
  %312 = xor i32 %310, %311
  %313 = shl i32 %310, 1
  %314 = xor i32 %313, 33800
  %315 = icmp slt i32 %312, 0
  %316 = select i1 %315, i32 %314, i32 %313
  %317 = shl i32 %31, 29
  %318 = xor i32 %316, %317
  %319 = shl i32 %316, 1
  %320 = xor i32 %319, 33800
  %321 = icmp slt i32 %318, 0
  %322 = select i1 %321, i32 %320, i32 %319
  %323 = shl i32 %31, 30
  %324 = xor i32 %322, %323
  %325 = shl i32 %322, 1
  %326 = xor i32 %325, 33800
  %327 = icmp slt i32 %324, 0
  %328 = select i1 %327, i32 %326, i32 %325
  %329 = shl i32 %31, 31
  %330 = xor i32 %328, %329
  %331 = shl i32 %328, 1
  %332 = xor i32 %331, 33800
  %333 = icmp slt i32 %330, 0
  %334 = select i1 %333, i32 %332, i32 %331
  %335 = icmp eq i32 %146, %334
  %336 = select i1 %335, i32 %26, i32 1
  %337 = add nuw nsw i64 %25, 1
  %338 = icmp eq i64 %337, 8
  br i1 %338, label %23, label %24, !llvm.loop !10
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
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
