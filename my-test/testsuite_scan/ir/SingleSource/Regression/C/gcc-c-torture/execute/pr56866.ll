; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr56866.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr56866.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [256 x i64], align 8
  %2 = alloca [256 x i64], align 8
  %3 = alloca [256 x i32], align 16
  %4 = alloca [256 x i32], align 16
  %5 = alloca [256 x i16], align 16
  %6 = alloca [256 x i16], align 16
  %7 = alloca [256 x i8], align 16
  %8 = alloca [256 x i8], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #5
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(2040) %9, i8 0, i64 2040, i1 false)
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(1020) %10, i8 0, i64 1020, i1 false)
  %11 = getelementptr inbounds nuw i8, ptr %5, i64 2
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(510) %11, i8 0, i64 510, i1 false)
  %12 = getelementptr inbounds nuw i8, ptr %7, i64 1
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(255) %12, i8 0, i64 255, i1 false)
  store i64 81985529216486895, ptr %1, align 8, !tbaa !6
  store i32 19088743, ptr %3, align 16, !tbaa !10
  store i16 17767, ptr %5, align 16, !tbaa !12
  store i8 115, ptr %7, align 16, !tbaa !14
  call void asm sideeffect "", "imr,imr,imr,imr,~{memory}"(ptr nonnull %1, ptr nonnull %3, ptr nonnull %5, ptr nonnull %7) #5, !srcloc !15
  br label %13

13:                                               ; preds = %13, %0
  %14 = phi i64 [ 0, %0 ], [ %23, %13 ]
  %15 = getelementptr inbounds nuw i64, ptr %1, i64 %14
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %17 = load <2 x i64>, ptr %15, align 8, !tbaa !6
  %18 = load <2 x i64>, ptr %16, align 8, !tbaa !6
  %19 = call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %17, <2 x i64> %17, <2 x i64> splat (i64 56))
  %20 = call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %18, <2 x i64> %18, <2 x i64> splat (i64 56))
  %21 = getelementptr inbounds nuw i64, ptr %2, i64 %14
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 16
  store <2 x i64> %19, ptr %21, align 8, !tbaa !6
  store <2 x i64> %20, ptr %22, align 8, !tbaa !6
  %23 = add nuw i64 %14, 4
  %24 = icmp eq i64 %23, 256
  br i1 %24, label %25, label %13, !llvm.loop !16

25:                                               ; preds = %13
  %26 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %27 = load <4 x i32>, ptr %3, align 16, !tbaa !10
  %28 = load <4 x i32>, ptr %26, align 16, !tbaa !10
  %29 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %27, <4 x i32> %27, <4 x i32> splat (i32 24))
  %30 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %28, <4 x i32> %28, <4 x i32> splat (i32 24))
  %31 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store <4 x i32> %29, ptr %4, align 16, !tbaa !10
  store <4 x i32> %30, ptr %31, align 16, !tbaa !10
  %32 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %33 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %34 = load <4 x i32>, ptr %32, align 16, !tbaa !10
  %35 = load <4 x i32>, ptr %33, align 16, !tbaa !10
  %36 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %34, <4 x i32> %34, <4 x i32> splat (i32 24))
  %37 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %35, <4 x i32> %35, <4 x i32> splat (i32 24))
  %38 = getelementptr inbounds nuw i8, ptr %4, i64 32
  %39 = getelementptr inbounds nuw i8, ptr %4, i64 48
  store <4 x i32> %36, ptr %38, align 16, !tbaa !10
  store <4 x i32> %37, ptr %39, align 16, !tbaa !10
  %40 = getelementptr inbounds nuw i8, ptr %3, i64 64
  %41 = getelementptr inbounds nuw i8, ptr %3, i64 80
  %42 = load <4 x i32>, ptr %40, align 16, !tbaa !10
  %43 = load <4 x i32>, ptr %41, align 16, !tbaa !10
  %44 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %42, <4 x i32> %42, <4 x i32> splat (i32 24))
  %45 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %43, <4 x i32> %43, <4 x i32> splat (i32 24))
  %46 = getelementptr inbounds nuw i8, ptr %4, i64 64
  %47 = getelementptr inbounds nuw i8, ptr %4, i64 80
  store <4 x i32> %44, ptr %46, align 16, !tbaa !10
  store <4 x i32> %45, ptr %47, align 16, !tbaa !10
  %48 = getelementptr inbounds nuw i8, ptr %3, i64 96
  %49 = getelementptr inbounds nuw i8, ptr %3, i64 112
  %50 = load <4 x i32>, ptr %48, align 16, !tbaa !10
  %51 = load <4 x i32>, ptr %49, align 16, !tbaa !10
  %52 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %50, <4 x i32> %50, <4 x i32> splat (i32 24))
  %53 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %51, <4 x i32> %51, <4 x i32> splat (i32 24))
  %54 = getelementptr inbounds nuw i8, ptr %4, i64 96
  %55 = getelementptr inbounds nuw i8, ptr %4, i64 112
  store <4 x i32> %52, ptr %54, align 16, !tbaa !10
  store <4 x i32> %53, ptr %55, align 16, !tbaa !10
  %56 = getelementptr inbounds nuw i8, ptr %3, i64 128
  %57 = getelementptr inbounds nuw i8, ptr %3, i64 144
  %58 = load <4 x i32>, ptr %56, align 16, !tbaa !10
  %59 = load <4 x i32>, ptr %57, align 16, !tbaa !10
  %60 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %58, <4 x i32> %58, <4 x i32> splat (i32 24))
  %61 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %59, <4 x i32> %59, <4 x i32> splat (i32 24))
  %62 = getelementptr inbounds nuw i8, ptr %4, i64 128
  %63 = getelementptr inbounds nuw i8, ptr %4, i64 144
  store <4 x i32> %60, ptr %62, align 16, !tbaa !10
  store <4 x i32> %61, ptr %63, align 16, !tbaa !10
  %64 = getelementptr inbounds nuw i8, ptr %3, i64 160
  %65 = getelementptr inbounds nuw i8, ptr %3, i64 176
  %66 = load <4 x i32>, ptr %64, align 16, !tbaa !10
  %67 = load <4 x i32>, ptr %65, align 16, !tbaa !10
  %68 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %66, <4 x i32> %66, <4 x i32> splat (i32 24))
  %69 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %67, <4 x i32> %67, <4 x i32> splat (i32 24))
  %70 = getelementptr inbounds nuw i8, ptr %4, i64 160
  %71 = getelementptr inbounds nuw i8, ptr %4, i64 176
  store <4 x i32> %68, ptr %70, align 16, !tbaa !10
  store <4 x i32> %69, ptr %71, align 16, !tbaa !10
  %72 = getelementptr inbounds nuw i8, ptr %3, i64 192
  %73 = getelementptr inbounds nuw i8, ptr %3, i64 208
  %74 = load <4 x i32>, ptr %72, align 16, !tbaa !10
  %75 = load <4 x i32>, ptr %73, align 16, !tbaa !10
  %76 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %74, <4 x i32> %74, <4 x i32> splat (i32 24))
  %77 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %75, <4 x i32> %75, <4 x i32> splat (i32 24))
  %78 = getelementptr inbounds nuw i8, ptr %4, i64 192
  %79 = getelementptr inbounds nuw i8, ptr %4, i64 208
  store <4 x i32> %76, ptr %78, align 16, !tbaa !10
  store <4 x i32> %77, ptr %79, align 16, !tbaa !10
  %80 = getelementptr inbounds nuw i8, ptr %3, i64 224
  %81 = getelementptr inbounds nuw i8, ptr %3, i64 240
  %82 = load <4 x i32>, ptr %80, align 16, !tbaa !10
  %83 = load <4 x i32>, ptr %81, align 16, !tbaa !10
  %84 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %82, <4 x i32> %82, <4 x i32> splat (i32 24))
  %85 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %83, <4 x i32> %83, <4 x i32> splat (i32 24))
  %86 = getelementptr inbounds nuw i8, ptr %4, i64 224
  %87 = getelementptr inbounds nuw i8, ptr %4, i64 240
  store <4 x i32> %84, ptr %86, align 16, !tbaa !10
  store <4 x i32> %85, ptr %87, align 16, !tbaa !10
  %88 = getelementptr inbounds nuw i8, ptr %3, i64 256
  %89 = getelementptr inbounds nuw i8, ptr %3, i64 272
  %90 = load <4 x i32>, ptr %88, align 16, !tbaa !10
  %91 = load <4 x i32>, ptr %89, align 16, !tbaa !10
  %92 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %90, <4 x i32> %90, <4 x i32> splat (i32 24))
  %93 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %91, <4 x i32> %91, <4 x i32> splat (i32 24))
  %94 = getelementptr inbounds nuw i8, ptr %4, i64 256
  %95 = getelementptr inbounds nuw i8, ptr %4, i64 272
  store <4 x i32> %92, ptr %94, align 16, !tbaa !10
  store <4 x i32> %93, ptr %95, align 16, !tbaa !10
  %96 = getelementptr inbounds nuw i8, ptr %3, i64 288
  %97 = getelementptr inbounds nuw i8, ptr %3, i64 304
  %98 = load <4 x i32>, ptr %96, align 16, !tbaa !10
  %99 = load <4 x i32>, ptr %97, align 16, !tbaa !10
  %100 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %98, <4 x i32> %98, <4 x i32> splat (i32 24))
  %101 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %99, <4 x i32> %99, <4 x i32> splat (i32 24))
  %102 = getelementptr inbounds nuw i8, ptr %4, i64 288
  %103 = getelementptr inbounds nuw i8, ptr %4, i64 304
  store <4 x i32> %100, ptr %102, align 16, !tbaa !10
  store <4 x i32> %101, ptr %103, align 16, !tbaa !10
  %104 = getelementptr inbounds nuw i8, ptr %3, i64 320
  %105 = getelementptr inbounds nuw i8, ptr %3, i64 336
  %106 = load <4 x i32>, ptr %104, align 16, !tbaa !10
  %107 = load <4 x i32>, ptr %105, align 16, !tbaa !10
  %108 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %106, <4 x i32> %106, <4 x i32> splat (i32 24))
  %109 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %107, <4 x i32> %107, <4 x i32> splat (i32 24))
  %110 = getelementptr inbounds nuw i8, ptr %4, i64 320
  %111 = getelementptr inbounds nuw i8, ptr %4, i64 336
  store <4 x i32> %108, ptr %110, align 16, !tbaa !10
  store <4 x i32> %109, ptr %111, align 16, !tbaa !10
  %112 = getelementptr inbounds nuw i8, ptr %3, i64 352
  %113 = getelementptr inbounds nuw i8, ptr %3, i64 368
  %114 = load <4 x i32>, ptr %112, align 16, !tbaa !10
  %115 = load <4 x i32>, ptr %113, align 16, !tbaa !10
  %116 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %114, <4 x i32> %114, <4 x i32> splat (i32 24))
  %117 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %115, <4 x i32> %115, <4 x i32> splat (i32 24))
  %118 = getelementptr inbounds nuw i8, ptr %4, i64 352
  %119 = getelementptr inbounds nuw i8, ptr %4, i64 368
  store <4 x i32> %116, ptr %118, align 16, !tbaa !10
  store <4 x i32> %117, ptr %119, align 16, !tbaa !10
  %120 = getelementptr inbounds nuw i8, ptr %3, i64 384
  %121 = getelementptr inbounds nuw i8, ptr %3, i64 400
  %122 = load <4 x i32>, ptr %120, align 16, !tbaa !10
  %123 = load <4 x i32>, ptr %121, align 16, !tbaa !10
  %124 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %122, <4 x i32> %122, <4 x i32> splat (i32 24))
  %125 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %123, <4 x i32> %123, <4 x i32> splat (i32 24))
  %126 = getelementptr inbounds nuw i8, ptr %4, i64 384
  %127 = getelementptr inbounds nuw i8, ptr %4, i64 400
  store <4 x i32> %124, ptr %126, align 16, !tbaa !10
  store <4 x i32> %125, ptr %127, align 16, !tbaa !10
  %128 = getelementptr inbounds nuw i8, ptr %3, i64 416
  %129 = getelementptr inbounds nuw i8, ptr %3, i64 432
  %130 = load <4 x i32>, ptr %128, align 16, !tbaa !10
  %131 = load <4 x i32>, ptr %129, align 16, !tbaa !10
  %132 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %130, <4 x i32> %130, <4 x i32> splat (i32 24))
  %133 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %131, <4 x i32> %131, <4 x i32> splat (i32 24))
  %134 = getelementptr inbounds nuw i8, ptr %4, i64 416
  %135 = getelementptr inbounds nuw i8, ptr %4, i64 432
  store <4 x i32> %132, ptr %134, align 16, !tbaa !10
  store <4 x i32> %133, ptr %135, align 16, !tbaa !10
  %136 = getelementptr inbounds nuw i8, ptr %3, i64 448
  %137 = getelementptr inbounds nuw i8, ptr %3, i64 464
  %138 = load <4 x i32>, ptr %136, align 16, !tbaa !10
  %139 = load <4 x i32>, ptr %137, align 16, !tbaa !10
  %140 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %138, <4 x i32> %138, <4 x i32> splat (i32 24))
  %141 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %139, <4 x i32> %139, <4 x i32> splat (i32 24))
  %142 = getelementptr inbounds nuw i8, ptr %4, i64 448
  %143 = getelementptr inbounds nuw i8, ptr %4, i64 464
  store <4 x i32> %140, ptr %142, align 16, !tbaa !10
  store <4 x i32> %141, ptr %143, align 16, !tbaa !10
  %144 = getelementptr inbounds nuw i8, ptr %3, i64 480
  %145 = getelementptr inbounds nuw i8, ptr %3, i64 496
  %146 = load <4 x i32>, ptr %144, align 16, !tbaa !10
  %147 = load <4 x i32>, ptr %145, align 16, !tbaa !10
  %148 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %146, <4 x i32> %146, <4 x i32> splat (i32 24))
  %149 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %147, <4 x i32> %147, <4 x i32> splat (i32 24))
  %150 = getelementptr inbounds nuw i8, ptr %4, i64 480
  %151 = getelementptr inbounds nuw i8, ptr %4, i64 496
  store <4 x i32> %148, ptr %150, align 16, !tbaa !10
  store <4 x i32> %149, ptr %151, align 16, !tbaa !10
  %152 = getelementptr inbounds nuw i8, ptr %3, i64 512
  %153 = getelementptr inbounds nuw i8, ptr %3, i64 528
  %154 = load <4 x i32>, ptr %152, align 16, !tbaa !10
  %155 = load <4 x i32>, ptr %153, align 16, !tbaa !10
  %156 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %154, <4 x i32> %154, <4 x i32> splat (i32 24))
  %157 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %155, <4 x i32> %155, <4 x i32> splat (i32 24))
  %158 = getelementptr inbounds nuw i8, ptr %4, i64 512
  %159 = getelementptr inbounds nuw i8, ptr %4, i64 528
  store <4 x i32> %156, ptr %158, align 16, !tbaa !10
  store <4 x i32> %157, ptr %159, align 16, !tbaa !10
  %160 = getelementptr inbounds nuw i8, ptr %3, i64 544
  %161 = getelementptr inbounds nuw i8, ptr %3, i64 560
  %162 = load <4 x i32>, ptr %160, align 16, !tbaa !10
  %163 = load <4 x i32>, ptr %161, align 16, !tbaa !10
  %164 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %162, <4 x i32> %162, <4 x i32> splat (i32 24))
  %165 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %163, <4 x i32> %163, <4 x i32> splat (i32 24))
  %166 = getelementptr inbounds nuw i8, ptr %4, i64 544
  %167 = getelementptr inbounds nuw i8, ptr %4, i64 560
  store <4 x i32> %164, ptr %166, align 16, !tbaa !10
  store <4 x i32> %165, ptr %167, align 16, !tbaa !10
  %168 = getelementptr inbounds nuw i8, ptr %3, i64 576
  %169 = getelementptr inbounds nuw i8, ptr %3, i64 592
  %170 = load <4 x i32>, ptr %168, align 16, !tbaa !10
  %171 = load <4 x i32>, ptr %169, align 16, !tbaa !10
  %172 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %170, <4 x i32> %170, <4 x i32> splat (i32 24))
  %173 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %171, <4 x i32> %171, <4 x i32> splat (i32 24))
  %174 = getelementptr inbounds nuw i8, ptr %4, i64 576
  %175 = getelementptr inbounds nuw i8, ptr %4, i64 592
  store <4 x i32> %172, ptr %174, align 16, !tbaa !10
  store <4 x i32> %173, ptr %175, align 16, !tbaa !10
  %176 = getelementptr inbounds nuw i8, ptr %3, i64 608
  %177 = getelementptr inbounds nuw i8, ptr %3, i64 624
  %178 = load <4 x i32>, ptr %176, align 16, !tbaa !10
  %179 = load <4 x i32>, ptr %177, align 16, !tbaa !10
  %180 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %178, <4 x i32> %178, <4 x i32> splat (i32 24))
  %181 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %179, <4 x i32> %179, <4 x i32> splat (i32 24))
  %182 = getelementptr inbounds nuw i8, ptr %4, i64 608
  %183 = getelementptr inbounds nuw i8, ptr %4, i64 624
  store <4 x i32> %180, ptr %182, align 16, !tbaa !10
  store <4 x i32> %181, ptr %183, align 16, !tbaa !10
  %184 = getelementptr inbounds nuw i8, ptr %3, i64 640
  %185 = getelementptr inbounds nuw i8, ptr %3, i64 656
  %186 = load <4 x i32>, ptr %184, align 16, !tbaa !10
  %187 = load <4 x i32>, ptr %185, align 16, !tbaa !10
  %188 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %186, <4 x i32> %186, <4 x i32> splat (i32 24))
  %189 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %187, <4 x i32> %187, <4 x i32> splat (i32 24))
  %190 = getelementptr inbounds nuw i8, ptr %4, i64 640
  %191 = getelementptr inbounds nuw i8, ptr %4, i64 656
  store <4 x i32> %188, ptr %190, align 16, !tbaa !10
  store <4 x i32> %189, ptr %191, align 16, !tbaa !10
  %192 = getelementptr inbounds nuw i8, ptr %3, i64 672
  %193 = getelementptr inbounds nuw i8, ptr %3, i64 688
  %194 = load <4 x i32>, ptr %192, align 16, !tbaa !10
  %195 = load <4 x i32>, ptr %193, align 16, !tbaa !10
  %196 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %194, <4 x i32> %194, <4 x i32> splat (i32 24))
  %197 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %195, <4 x i32> %195, <4 x i32> splat (i32 24))
  %198 = getelementptr inbounds nuw i8, ptr %4, i64 672
  %199 = getelementptr inbounds nuw i8, ptr %4, i64 688
  store <4 x i32> %196, ptr %198, align 16, !tbaa !10
  store <4 x i32> %197, ptr %199, align 16, !tbaa !10
  %200 = getelementptr inbounds nuw i8, ptr %3, i64 704
  %201 = getelementptr inbounds nuw i8, ptr %3, i64 720
  %202 = load <4 x i32>, ptr %200, align 16, !tbaa !10
  %203 = load <4 x i32>, ptr %201, align 16, !tbaa !10
  %204 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %202, <4 x i32> %202, <4 x i32> splat (i32 24))
  %205 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %203, <4 x i32> %203, <4 x i32> splat (i32 24))
  %206 = getelementptr inbounds nuw i8, ptr %4, i64 704
  %207 = getelementptr inbounds nuw i8, ptr %4, i64 720
  store <4 x i32> %204, ptr %206, align 16, !tbaa !10
  store <4 x i32> %205, ptr %207, align 16, !tbaa !10
  %208 = getelementptr inbounds nuw i8, ptr %3, i64 736
  %209 = getelementptr inbounds nuw i8, ptr %3, i64 752
  %210 = load <4 x i32>, ptr %208, align 16, !tbaa !10
  %211 = load <4 x i32>, ptr %209, align 16, !tbaa !10
  %212 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %210, <4 x i32> %210, <4 x i32> splat (i32 24))
  %213 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %211, <4 x i32> %211, <4 x i32> splat (i32 24))
  %214 = getelementptr inbounds nuw i8, ptr %4, i64 736
  %215 = getelementptr inbounds nuw i8, ptr %4, i64 752
  store <4 x i32> %212, ptr %214, align 16, !tbaa !10
  store <4 x i32> %213, ptr %215, align 16, !tbaa !10
  %216 = getelementptr inbounds nuw i8, ptr %3, i64 768
  %217 = getelementptr inbounds nuw i8, ptr %3, i64 784
  %218 = load <4 x i32>, ptr %216, align 16, !tbaa !10
  %219 = load <4 x i32>, ptr %217, align 16, !tbaa !10
  %220 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %218, <4 x i32> %218, <4 x i32> splat (i32 24))
  %221 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %219, <4 x i32> %219, <4 x i32> splat (i32 24))
  %222 = getelementptr inbounds nuw i8, ptr %4, i64 768
  %223 = getelementptr inbounds nuw i8, ptr %4, i64 784
  store <4 x i32> %220, ptr %222, align 16, !tbaa !10
  store <4 x i32> %221, ptr %223, align 16, !tbaa !10
  %224 = getelementptr inbounds nuw i8, ptr %3, i64 800
  %225 = getelementptr inbounds nuw i8, ptr %3, i64 816
  %226 = load <4 x i32>, ptr %224, align 16, !tbaa !10
  %227 = load <4 x i32>, ptr %225, align 16, !tbaa !10
  %228 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %226, <4 x i32> %226, <4 x i32> splat (i32 24))
  %229 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %227, <4 x i32> %227, <4 x i32> splat (i32 24))
  %230 = getelementptr inbounds nuw i8, ptr %4, i64 800
  %231 = getelementptr inbounds nuw i8, ptr %4, i64 816
  store <4 x i32> %228, ptr %230, align 16, !tbaa !10
  store <4 x i32> %229, ptr %231, align 16, !tbaa !10
  %232 = getelementptr inbounds nuw i8, ptr %3, i64 832
  %233 = getelementptr inbounds nuw i8, ptr %3, i64 848
  %234 = load <4 x i32>, ptr %232, align 16, !tbaa !10
  %235 = load <4 x i32>, ptr %233, align 16, !tbaa !10
  %236 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %234, <4 x i32> %234, <4 x i32> splat (i32 24))
  %237 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %235, <4 x i32> %235, <4 x i32> splat (i32 24))
  %238 = getelementptr inbounds nuw i8, ptr %4, i64 832
  %239 = getelementptr inbounds nuw i8, ptr %4, i64 848
  store <4 x i32> %236, ptr %238, align 16, !tbaa !10
  store <4 x i32> %237, ptr %239, align 16, !tbaa !10
  %240 = getelementptr inbounds nuw i8, ptr %3, i64 864
  %241 = getelementptr inbounds nuw i8, ptr %3, i64 880
  %242 = load <4 x i32>, ptr %240, align 16, !tbaa !10
  %243 = load <4 x i32>, ptr %241, align 16, !tbaa !10
  %244 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %242, <4 x i32> %242, <4 x i32> splat (i32 24))
  %245 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %243, <4 x i32> %243, <4 x i32> splat (i32 24))
  %246 = getelementptr inbounds nuw i8, ptr %4, i64 864
  %247 = getelementptr inbounds nuw i8, ptr %4, i64 880
  store <4 x i32> %244, ptr %246, align 16, !tbaa !10
  store <4 x i32> %245, ptr %247, align 16, !tbaa !10
  %248 = getelementptr inbounds nuw i8, ptr %3, i64 896
  %249 = getelementptr inbounds nuw i8, ptr %3, i64 912
  %250 = load <4 x i32>, ptr %248, align 16, !tbaa !10
  %251 = load <4 x i32>, ptr %249, align 16, !tbaa !10
  %252 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %250, <4 x i32> %250, <4 x i32> splat (i32 24))
  %253 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %251, <4 x i32> %251, <4 x i32> splat (i32 24))
  %254 = getelementptr inbounds nuw i8, ptr %4, i64 896
  %255 = getelementptr inbounds nuw i8, ptr %4, i64 912
  store <4 x i32> %252, ptr %254, align 16, !tbaa !10
  store <4 x i32> %253, ptr %255, align 16, !tbaa !10
  %256 = getelementptr inbounds nuw i8, ptr %3, i64 928
  %257 = getelementptr inbounds nuw i8, ptr %3, i64 944
  %258 = load <4 x i32>, ptr %256, align 16, !tbaa !10
  %259 = load <4 x i32>, ptr %257, align 16, !tbaa !10
  %260 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %258, <4 x i32> %258, <4 x i32> splat (i32 24))
  %261 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %259, <4 x i32> %259, <4 x i32> splat (i32 24))
  %262 = getelementptr inbounds nuw i8, ptr %4, i64 928
  %263 = getelementptr inbounds nuw i8, ptr %4, i64 944
  store <4 x i32> %260, ptr %262, align 16, !tbaa !10
  store <4 x i32> %261, ptr %263, align 16, !tbaa !10
  %264 = getelementptr inbounds nuw i8, ptr %3, i64 960
  %265 = getelementptr inbounds nuw i8, ptr %3, i64 976
  %266 = load <4 x i32>, ptr %264, align 16, !tbaa !10
  %267 = load <4 x i32>, ptr %265, align 16, !tbaa !10
  %268 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %266, <4 x i32> %266, <4 x i32> splat (i32 24))
  %269 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %267, <4 x i32> %267, <4 x i32> splat (i32 24))
  %270 = getelementptr inbounds nuw i8, ptr %4, i64 960
  %271 = getelementptr inbounds nuw i8, ptr %4, i64 976
  store <4 x i32> %268, ptr %270, align 16, !tbaa !10
  store <4 x i32> %269, ptr %271, align 16, !tbaa !10
  %272 = getelementptr inbounds nuw i8, ptr %3, i64 992
  %273 = getelementptr inbounds nuw i8, ptr %3, i64 1008
  %274 = load <4 x i32>, ptr %272, align 16, !tbaa !10
  %275 = load <4 x i32>, ptr %273, align 16, !tbaa !10
  %276 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %274, <4 x i32> %274, <4 x i32> splat (i32 24))
  %277 = call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %275, <4 x i32> %275, <4 x i32> splat (i32 24))
  %278 = getelementptr inbounds nuw i8, ptr %4, i64 992
  %279 = getelementptr inbounds nuw i8, ptr %4, i64 1008
  store <4 x i32> %276, ptr %278, align 16, !tbaa !10
  store <4 x i32> %277, ptr %279, align 16, !tbaa !10
  %280 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %281 = load <8 x i16>, ptr %5, align 16, !tbaa !12
  %282 = load <8 x i16>, ptr %280, align 16, !tbaa !12
  %283 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %281, <8 x i16> %281, <8 x i16> splat (i16 7))
  %284 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %282, <8 x i16> %282, <8 x i16> splat (i16 7))
  %285 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store <8 x i16> %283, ptr %6, align 16, !tbaa !12
  store <8 x i16> %284, ptr %285, align 16, !tbaa !12
  %286 = getelementptr inbounds nuw i8, ptr %5, i64 32
  %287 = getelementptr inbounds nuw i8, ptr %5, i64 48
  %288 = load <8 x i16>, ptr %286, align 16, !tbaa !12
  %289 = load <8 x i16>, ptr %287, align 16, !tbaa !12
  %290 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %288, <8 x i16> %288, <8 x i16> splat (i16 7))
  %291 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %289, <8 x i16> %289, <8 x i16> splat (i16 7))
  %292 = getelementptr inbounds nuw i8, ptr %6, i64 32
  %293 = getelementptr inbounds nuw i8, ptr %6, i64 48
  store <8 x i16> %290, ptr %292, align 16, !tbaa !12
  store <8 x i16> %291, ptr %293, align 16, !tbaa !12
  %294 = getelementptr inbounds nuw i8, ptr %5, i64 64
  %295 = getelementptr inbounds nuw i8, ptr %5, i64 80
  %296 = load <8 x i16>, ptr %294, align 16, !tbaa !12
  %297 = load <8 x i16>, ptr %295, align 16, !tbaa !12
  %298 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %296, <8 x i16> %296, <8 x i16> splat (i16 7))
  %299 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %297, <8 x i16> %297, <8 x i16> splat (i16 7))
  %300 = getelementptr inbounds nuw i8, ptr %6, i64 64
  %301 = getelementptr inbounds nuw i8, ptr %6, i64 80
  store <8 x i16> %298, ptr %300, align 16, !tbaa !12
  store <8 x i16> %299, ptr %301, align 16, !tbaa !12
  %302 = getelementptr inbounds nuw i8, ptr %5, i64 96
  %303 = getelementptr inbounds nuw i8, ptr %5, i64 112
  %304 = load <8 x i16>, ptr %302, align 16, !tbaa !12
  %305 = load <8 x i16>, ptr %303, align 16, !tbaa !12
  %306 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %304, <8 x i16> %304, <8 x i16> splat (i16 7))
  %307 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %305, <8 x i16> %305, <8 x i16> splat (i16 7))
  %308 = getelementptr inbounds nuw i8, ptr %6, i64 96
  %309 = getelementptr inbounds nuw i8, ptr %6, i64 112
  store <8 x i16> %306, ptr %308, align 16, !tbaa !12
  store <8 x i16> %307, ptr %309, align 16, !tbaa !12
  %310 = getelementptr inbounds nuw i8, ptr %5, i64 128
  %311 = getelementptr inbounds nuw i8, ptr %5, i64 144
  %312 = load <8 x i16>, ptr %310, align 16, !tbaa !12
  %313 = load <8 x i16>, ptr %311, align 16, !tbaa !12
  %314 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %312, <8 x i16> %312, <8 x i16> splat (i16 7))
  %315 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %313, <8 x i16> %313, <8 x i16> splat (i16 7))
  %316 = getelementptr inbounds nuw i8, ptr %6, i64 128
  %317 = getelementptr inbounds nuw i8, ptr %6, i64 144
  store <8 x i16> %314, ptr %316, align 16, !tbaa !12
  store <8 x i16> %315, ptr %317, align 16, !tbaa !12
  %318 = getelementptr inbounds nuw i8, ptr %5, i64 160
  %319 = getelementptr inbounds nuw i8, ptr %5, i64 176
  %320 = load <8 x i16>, ptr %318, align 16, !tbaa !12
  %321 = load <8 x i16>, ptr %319, align 16, !tbaa !12
  %322 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %320, <8 x i16> %320, <8 x i16> splat (i16 7))
  %323 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %321, <8 x i16> %321, <8 x i16> splat (i16 7))
  %324 = getelementptr inbounds nuw i8, ptr %6, i64 160
  %325 = getelementptr inbounds nuw i8, ptr %6, i64 176
  store <8 x i16> %322, ptr %324, align 16, !tbaa !12
  store <8 x i16> %323, ptr %325, align 16, !tbaa !12
  %326 = getelementptr inbounds nuw i8, ptr %5, i64 192
  %327 = getelementptr inbounds nuw i8, ptr %5, i64 208
  %328 = load <8 x i16>, ptr %326, align 16, !tbaa !12
  %329 = load <8 x i16>, ptr %327, align 16, !tbaa !12
  %330 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %328, <8 x i16> %328, <8 x i16> splat (i16 7))
  %331 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %329, <8 x i16> %329, <8 x i16> splat (i16 7))
  %332 = getelementptr inbounds nuw i8, ptr %6, i64 192
  %333 = getelementptr inbounds nuw i8, ptr %6, i64 208
  store <8 x i16> %330, ptr %332, align 16, !tbaa !12
  store <8 x i16> %331, ptr %333, align 16, !tbaa !12
  %334 = getelementptr inbounds nuw i8, ptr %5, i64 224
  %335 = getelementptr inbounds nuw i8, ptr %5, i64 240
  %336 = load <8 x i16>, ptr %334, align 16, !tbaa !12
  %337 = load <8 x i16>, ptr %335, align 16, !tbaa !12
  %338 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %336, <8 x i16> %336, <8 x i16> splat (i16 7))
  %339 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %337, <8 x i16> %337, <8 x i16> splat (i16 7))
  %340 = getelementptr inbounds nuw i8, ptr %6, i64 224
  %341 = getelementptr inbounds nuw i8, ptr %6, i64 240
  store <8 x i16> %338, ptr %340, align 16, !tbaa !12
  store <8 x i16> %339, ptr %341, align 16, !tbaa !12
  %342 = getelementptr inbounds nuw i8, ptr %5, i64 256
  %343 = getelementptr inbounds nuw i8, ptr %5, i64 272
  %344 = load <8 x i16>, ptr %342, align 16, !tbaa !12
  %345 = load <8 x i16>, ptr %343, align 16, !tbaa !12
  %346 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %344, <8 x i16> %344, <8 x i16> splat (i16 7))
  %347 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %345, <8 x i16> %345, <8 x i16> splat (i16 7))
  %348 = getelementptr inbounds nuw i8, ptr %6, i64 256
  %349 = getelementptr inbounds nuw i8, ptr %6, i64 272
  store <8 x i16> %346, ptr %348, align 16, !tbaa !12
  store <8 x i16> %347, ptr %349, align 16, !tbaa !12
  %350 = getelementptr inbounds nuw i8, ptr %5, i64 288
  %351 = getelementptr inbounds nuw i8, ptr %5, i64 304
  %352 = load <8 x i16>, ptr %350, align 16, !tbaa !12
  %353 = load <8 x i16>, ptr %351, align 16, !tbaa !12
  %354 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %352, <8 x i16> %352, <8 x i16> splat (i16 7))
  %355 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %353, <8 x i16> %353, <8 x i16> splat (i16 7))
  %356 = getelementptr inbounds nuw i8, ptr %6, i64 288
  %357 = getelementptr inbounds nuw i8, ptr %6, i64 304
  store <8 x i16> %354, ptr %356, align 16, !tbaa !12
  store <8 x i16> %355, ptr %357, align 16, !tbaa !12
  %358 = getelementptr inbounds nuw i8, ptr %5, i64 320
  %359 = getelementptr inbounds nuw i8, ptr %5, i64 336
  %360 = load <8 x i16>, ptr %358, align 16, !tbaa !12
  %361 = load <8 x i16>, ptr %359, align 16, !tbaa !12
  %362 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %360, <8 x i16> %360, <8 x i16> splat (i16 7))
  %363 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %361, <8 x i16> %361, <8 x i16> splat (i16 7))
  %364 = getelementptr inbounds nuw i8, ptr %6, i64 320
  %365 = getelementptr inbounds nuw i8, ptr %6, i64 336
  store <8 x i16> %362, ptr %364, align 16, !tbaa !12
  store <8 x i16> %363, ptr %365, align 16, !tbaa !12
  %366 = getelementptr inbounds nuw i8, ptr %5, i64 352
  %367 = getelementptr inbounds nuw i8, ptr %5, i64 368
  %368 = load <8 x i16>, ptr %366, align 16, !tbaa !12
  %369 = load <8 x i16>, ptr %367, align 16, !tbaa !12
  %370 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %368, <8 x i16> %368, <8 x i16> splat (i16 7))
  %371 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %369, <8 x i16> %369, <8 x i16> splat (i16 7))
  %372 = getelementptr inbounds nuw i8, ptr %6, i64 352
  %373 = getelementptr inbounds nuw i8, ptr %6, i64 368
  store <8 x i16> %370, ptr %372, align 16, !tbaa !12
  store <8 x i16> %371, ptr %373, align 16, !tbaa !12
  %374 = getelementptr inbounds nuw i8, ptr %5, i64 384
  %375 = getelementptr inbounds nuw i8, ptr %5, i64 400
  %376 = load <8 x i16>, ptr %374, align 16, !tbaa !12
  %377 = load <8 x i16>, ptr %375, align 16, !tbaa !12
  %378 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %376, <8 x i16> %376, <8 x i16> splat (i16 7))
  %379 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %377, <8 x i16> %377, <8 x i16> splat (i16 7))
  %380 = getelementptr inbounds nuw i8, ptr %6, i64 384
  %381 = getelementptr inbounds nuw i8, ptr %6, i64 400
  store <8 x i16> %378, ptr %380, align 16, !tbaa !12
  store <8 x i16> %379, ptr %381, align 16, !tbaa !12
  %382 = getelementptr inbounds nuw i8, ptr %5, i64 416
  %383 = getelementptr inbounds nuw i8, ptr %5, i64 432
  %384 = load <8 x i16>, ptr %382, align 16, !tbaa !12
  %385 = load <8 x i16>, ptr %383, align 16, !tbaa !12
  %386 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %384, <8 x i16> %384, <8 x i16> splat (i16 7))
  %387 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %385, <8 x i16> %385, <8 x i16> splat (i16 7))
  %388 = getelementptr inbounds nuw i8, ptr %6, i64 416
  %389 = getelementptr inbounds nuw i8, ptr %6, i64 432
  store <8 x i16> %386, ptr %388, align 16, !tbaa !12
  store <8 x i16> %387, ptr %389, align 16, !tbaa !12
  %390 = getelementptr inbounds nuw i8, ptr %5, i64 448
  %391 = getelementptr inbounds nuw i8, ptr %5, i64 464
  %392 = load <8 x i16>, ptr %390, align 16, !tbaa !12
  %393 = load <8 x i16>, ptr %391, align 16, !tbaa !12
  %394 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %392, <8 x i16> %392, <8 x i16> splat (i16 7))
  %395 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %393, <8 x i16> %393, <8 x i16> splat (i16 7))
  %396 = getelementptr inbounds nuw i8, ptr %6, i64 448
  %397 = getelementptr inbounds nuw i8, ptr %6, i64 464
  store <8 x i16> %394, ptr %396, align 16, !tbaa !12
  store <8 x i16> %395, ptr %397, align 16, !tbaa !12
  %398 = getelementptr inbounds nuw i8, ptr %5, i64 480
  %399 = getelementptr inbounds nuw i8, ptr %5, i64 496
  %400 = load <8 x i16>, ptr %398, align 16, !tbaa !12
  %401 = load <8 x i16>, ptr %399, align 16, !tbaa !12
  %402 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %400, <8 x i16> %400, <8 x i16> splat (i16 7))
  %403 = call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %401, <8 x i16> %401, <8 x i16> splat (i16 7))
  %404 = getelementptr inbounds nuw i8, ptr %6, i64 480
  %405 = getelementptr inbounds nuw i8, ptr %6, i64 496
  store <8 x i16> %402, ptr %404, align 16, !tbaa !12
  store <8 x i16> %403, ptr %405, align 16, !tbaa !12
  %406 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %407 = load <16 x i8>, ptr %7, align 16, !tbaa !14
  %408 = load <16 x i8>, ptr %406, align 16, !tbaa !14
  %409 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %407, <16 x i8> %407, <16 x i8> splat (i8 3))
  %410 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %408, <16 x i8> %408, <16 x i8> splat (i8 3))
  %411 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store <16 x i8> %409, ptr %8, align 16, !tbaa !14
  store <16 x i8> %410, ptr %411, align 16, !tbaa !14
  %412 = getelementptr inbounds nuw i8, ptr %7, i64 32
  %413 = getelementptr inbounds nuw i8, ptr %7, i64 48
  %414 = load <16 x i8>, ptr %412, align 16, !tbaa !14
  %415 = load <16 x i8>, ptr %413, align 16, !tbaa !14
  %416 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %414, <16 x i8> %414, <16 x i8> splat (i8 3))
  %417 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %415, <16 x i8> %415, <16 x i8> splat (i8 3))
  %418 = getelementptr inbounds nuw i8, ptr %8, i64 32
  %419 = getelementptr inbounds nuw i8, ptr %8, i64 48
  store <16 x i8> %416, ptr %418, align 16, !tbaa !14
  store <16 x i8> %417, ptr %419, align 16, !tbaa !14
  %420 = getelementptr inbounds nuw i8, ptr %7, i64 64
  %421 = getelementptr inbounds nuw i8, ptr %7, i64 80
  %422 = load <16 x i8>, ptr %420, align 16, !tbaa !14
  %423 = load <16 x i8>, ptr %421, align 16, !tbaa !14
  %424 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %422, <16 x i8> %422, <16 x i8> splat (i8 3))
  %425 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %423, <16 x i8> %423, <16 x i8> splat (i8 3))
  %426 = getelementptr inbounds nuw i8, ptr %8, i64 64
  %427 = getelementptr inbounds nuw i8, ptr %8, i64 80
  store <16 x i8> %424, ptr %426, align 16, !tbaa !14
  store <16 x i8> %425, ptr %427, align 16, !tbaa !14
  %428 = getelementptr inbounds nuw i8, ptr %7, i64 96
  %429 = getelementptr inbounds nuw i8, ptr %7, i64 112
  %430 = load <16 x i8>, ptr %428, align 16, !tbaa !14
  %431 = load <16 x i8>, ptr %429, align 16, !tbaa !14
  %432 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %430, <16 x i8> %430, <16 x i8> splat (i8 3))
  %433 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %431, <16 x i8> %431, <16 x i8> splat (i8 3))
  %434 = getelementptr inbounds nuw i8, ptr %8, i64 96
  %435 = getelementptr inbounds nuw i8, ptr %8, i64 112
  store <16 x i8> %432, ptr %434, align 16, !tbaa !14
  store <16 x i8> %433, ptr %435, align 16, !tbaa !14
  %436 = getelementptr inbounds nuw i8, ptr %7, i64 128
  %437 = getelementptr inbounds nuw i8, ptr %7, i64 144
  %438 = load <16 x i8>, ptr %436, align 16, !tbaa !14
  %439 = load <16 x i8>, ptr %437, align 16, !tbaa !14
  %440 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %438, <16 x i8> %438, <16 x i8> splat (i8 3))
  %441 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %439, <16 x i8> %439, <16 x i8> splat (i8 3))
  %442 = getelementptr inbounds nuw i8, ptr %8, i64 128
  %443 = getelementptr inbounds nuw i8, ptr %8, i64 144
  store <16 x i8> %440, ptr %442, align 16, !tbaa !14
  store <16 x i8> %441, ptr %443, align 16, !tbaa !14
  %444 = getelementptr inbounds nuw i8, ptr %7, i64 160
  %445 = getelementptr inbounds nuw i8, ptr %7, i64 176
  %446 = load <16 x i8>, ptr %444, align 16, !tbaa !14
  %447 = load <16 x i8>, ptr %445, align 16, !tbaa !14
  %448 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %446, <16 x i8> %446, <16 x i8> splat (i8 3))
  %449 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %447, <16 x i8> %447, <16 x i8> splat (i8 3))
  %450 = getelementptr inbounds nuw i8, ptr %8, i64 160
  %451 = getelementptr inbounds nuw i8, ptr %8, i64 176
  store <16 x i8> %448, ptr %450, align 16, !tbaa !14
  store <16 x i8> %449, ptr %451, align 16, !tbaa !14
  %452 = getelementptr inbounds nuw i8, ptr %7, i64 192
  %453 = getelementptr inbounds nuw i8, ptr %7, i64 208
  %454 = load <16 x i8>, ptr %452, align 16, !tbaa !14
  %455 = load <16 x i8>, ptr %453, align 16, !tbaa !14
  %456 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %454, <16 x i8> %454, <16 x i8> splat (i8 3))
  %457 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %455, <16 x i8> %455, <16 x i8> splat (i8 3))
  %458 = getelementptr inbounds nuw i8, ptr %8, i64 192
  %459 = getelementptr inbounds nuw i8, ptr %8, i64 208
  store <16 x i8> %456, ptr %458, align 16, !tbaa !14
  store <16 x i8> %457, ptr %459, align 16, !tbaa !14
  %460 = getelementptr inbounds nuw i8, ptr %7, i64 224
  %461 = getelementptr inbounds nuw i8, ptr %7, i64 240
  %462 = load <16 x i8>, ptr %460, align 16, !tbaa !14
  %463 = load <16 x i8>, ptr %461, align 16, !tbaa !14
  %464 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %462, <16 x i8> %462, <16 x i8> splat (i8 3))
  %465 = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %463, <16 x i8> %463, <16 x i8> splat (i8 3))
  %466 = getelementptr inbounds nuw i8, ptr %8, i64 224
  %467 = getelementptr inbounds nuw i8, ptr %8, i64 240
  store <16 x i8> %464, ptr %466, align 16, !tbaa !14
  store <16 x i8> %465, ptr %467, align 16, !tbaa !14
  call void asm sideeffect "", "imr,imr,imr,imr,~{memory}"(ptr nonnull %2, ptr nonnull %4, ptr nonnull %6, ptr nonnull %8) #5, !srcloc !20
  %468 = load i64, ptr %2, align 8, !tbaa !6
  %469 = icmp ne i64 %468, -1224658842671273011
  %470 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %471 = load i64, ptr %470, align 8
  %472 = icmp ne i64 %471, 0
  %473 = select i1 %469, i1 true, i1 %472
  br i1 %473, label %474, label %475

474:                                              ; preds = %25
  call void @abort() #6
  unreachable

475:                                              ; preds = %25
  %476 = load i32, ptr %4, align 16, !tbaa !10
  %477 = icmp ne i32 %476, 1728127813
  %478 = getelementptr inbounds nuw i8, ptr %4, i64 4
  %479 = load i32, ptr %478, align 4
  %480 = icmp ne i32 %479, 0
  %481 = select i1 %477, i1 true, i1 %480
  br i1 %481, label %482, label %483

482:                                              ; preds = %475
  call void @abort() #6
  unreachable

483:                                              ; preds = %475
  %484 = load i16, ptr %6, align 16, !tbaa !12
  %485 = icmp ne i16 %484, -19550
  %486 = getelementptr inbounds nuw i8, ptr %6, i64 2
  %487 = load i16, ptr %486, align 2
  %488 = icmp ne i16 %487, 0
  %489 = select i1 %485, i1 true, i1 %488
  br i1 %489, label %490, label %491

490:                                              ; preds = %483
  call void @abort() #6
  unreachable

491:                                              ; preds = %483
  %492 = load i8, ptr %8, align 16, !tbaa !14
  %493 = icmp ne i8 %492, -101
  %494 = getelementptr inbounds nuw i8, ptr %8, i64 1
  %495 = load i8, ptr %494, align 1
  %496 = icmp ne i8 %495, 0
  %497 = select i1 %493, i1 true, i1 %496
  br i1 %497, label %498, label %499

498:                                              ; preds = %491
  call void @abort() #6
  unreachable

499:                                              ; preds = %491
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i64> @llvm.fshl.v2i64(<2 x i64>, <2 x i64>, <2 x i64>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x i32> @llvm.fshl.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x i16> @llvm.fshl.v8i16(<8 x i16>, <8 x i16>, <8 x i16>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x i8> @llvm.fshl.v16i8(<16 x i8>, <16 x i8>, <16 x i8>) #4

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"short", !8, i64 0}
!14 = !{!8, !8, i64 0}
!15 = !{i64 551}
!16 = distinct !{!16, !17, !18, !19}
!17 = !{!"llvm.loop.mustprogress"}
!18 = !{!"llvm.loop.isvectorized", i32 1}
!19 = !{!"llvm.loop.unroll.runtime.disable"}
!20 = !{i64 1040}
