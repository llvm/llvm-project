; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/test_indvars.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/test_indvars.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [18 x i8] c"Checksum = %.0lf\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [100 x [200 x i32]], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(788) %2, i8 0, i64 788, i1 false), !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 1600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %3, i8 0, i64 800, i1 false), !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 3200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %4, i8 0, i64 800, i1 false), !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 4800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %5, i8 0, i64 800, i1 false), !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 6400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %6, i8 0, i64 800, i1 false), !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %7, i8 0, i64 800, i1 false), !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 9600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %8, i8 0, i64 800, i1 false), !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 11200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %9, i8 0, i64 800, i1 false), !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 12800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %10, i8 0, i64 800, i1 false), !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 14400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %11, i8 0, i64 800, i1 false), !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %12, i8 0, i64 800, i1 false), !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 17600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %13, i8 0, i64 800, i1 false), !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 19200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %14, i8 0, i64 800, i1 false), !tbaa !6
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 20800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %15, i8 0, i64 800, i1 false), !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 22400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %16, i8 0, i64 800, i1 false), !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 24000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %17, i8 0, i64 800, i1 false), !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 25600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %18, i8 0, i64 800, i1 false), !tbaa !6
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 27200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %19, i8 0, i64 800, i1 false), !tbaa !6
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 28800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %20, i8 0, i64 800, i1 false), !tbaa !6
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 30400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %21, i8 0, i64 800, i1 false), !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 32000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %22, i8 0, i64 800, i1 false), !tbaa !6
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 33600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %23, i8 0, i64 800, i1 false), !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 35200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %24, i8 0, i64 800, i1 false), !tbaa !6
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 36800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %25, i8 0, i64 800, i1 false), !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 38400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %26, i8 0, i64 800, i1 false), !tbaa !6
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 40000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %27, i8 0, i64 800, i1 false), !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 41600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %28, i8 0, i64 800, i1 false), !tbaa !6
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 43200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %29, i8 0, i64 800, i1 false), !tbaa !6
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 44800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %30, i8 0, i64 800, i1 false), !tbaa !6
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 46400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %31, i8 0, i64 800, i1 false), !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 48000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %32, i8 0, i64 800, i1 false), !tbaa !6
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 49600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %33, i8 0, i64 800, i1 false), !tbaa !6
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 51200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %34, i8 0, i64 800, i1 false), !tbaa !6
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 52800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %35, i8 0, i64 800, i1 false), !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 54400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %36, i8 0, i64 800, i1 false), !tbaa !6
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 56000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %37, i8 0, i64 800, i1 false), !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 57600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %38, i8 0, i64 800, i1 false), !tbaa !6
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 59200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %39, i8 0, i64 800, i1 false), !tbaa !6
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 60800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %40, i8 0, i64 800, i1 false), !tbaa !6
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 62400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %41, i8 0, i64 800, i1 false), !tbaa !6
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 64000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %42, i8 0, i64 800, i1 false), !tbaa !6
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 65600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %43, i8 0, i64 800, i1 false), !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %1, i64 67200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %44, i8 0, i64 800, i1 false), !tbaa !6
  %45 = getelementptr inbounds nuw i8, ptr %1, i64 68800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %45, i8 0, i64 800, i1 false), !tbaa !6
  %46 = getelementptr inbounds nuw i8, ptr %1, i64 70400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %46, i8 0, i64 800, i1 false), !tbaa !6
  %47 = getelementptr inbounds nuw i8, ptr %1, i64 72000
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %47, i8 0, i64 800, i1 false), !tbaa !6
  %48 = getelementptr inbounds nuw i8, ptr %1, i64 73600
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %48, i8 0, i64 800, i1 false), !tbaa !6
  %49 = getelementptr inbounds nuw i8, ptr %1, i64 75200
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %49, i8 0, i64 800, i1 false), !tbaa !6
  %50 = getelementptr inbounds nuw i8, ptr %1, i64 76800
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %50, i8 0, i64 800, i1 false), !tbaa !6
  %51 = getelementptr inbounds nuw i8, ptr %1, i64 78400
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(800) %51, i8 0, i64 800, i1 false), !tbaa !6
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 2424
  store i32 12345, ptr %52, align 8, !tbaa !6
  store <2 x i32> <i32 0, i32 12345>, ptr %1, align 8, !tbaa !6
  %53 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 2, ptr %53, align 8, !tbaa !6
  %54 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i32 4, ptr %54, align 8, !tbaa !6
  %55 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i32 6, ptr %55, align 8, !tbaa !6
  %56 = getelementptr inbounds nuw i8, ptr %1, i64 32
  store i32 8, ptr %56, align 8, !tbaa !6
  %57 = getelementptr inbounds nuw i8, ptr %1, i64 40
  store i32 10, ptr %57, align 8, !tbaa !6
  %58 = getelementptr inbounds nuw i8, ptr %1, i64 48
  store i32 12, ptr %58, align 8, !tbaa !6
  %59 = getelementptr inbounds nuw i8, ptr %1, i64 56
  store i32 14, ptr %59, align 8, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %1, i64 64
  store i32 16, ptr %60, align 8, !tbaa !6
  %61 = getelementptr inbounds nuw i8, ptr %1, i64 72
  store i32 18, ptr %61, align 8, !tbaa !6
  %62 = getelementptr inbounds nuw i8, ptr %1, i64 80
  store i32 20, ptr %62, align 8, !tbaa !6
  %63 = getelementptr inbounds nuw i8, ptr %1, i64 88
  store i32 22, ptr %63, align 8, !tbaa !6
  %64 = getelementptr inbounds nuw i8, ptr %1, i64 96
  store i32 24, ptr %64, align 8, !tbaa !6
  %65 = getelementptr inbounds nuw i8, ptr %1, i64 104
  store i32 26, ptr %65, align 8, !tbaa !6
  %66 = getelementptr inbounds nuw i8, ptr %1, i64 112
  store i32 28, ptr %66, align 8, !tbaa !6
  %67 = getelementptr inbounds nuw i8, ptr %1, i64 120
  store i32 30, ptr %67, align 8, !tbaa !6
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 128
  store i32 32, ptr %68, align 8, !tbaa !6
  %69 = getelementptr inbounds nuw i8, ptr %1, i64 136
  store i32 34, ptr %69, align 8, !tbaa !6
  %70 = getelementptr inbounds nuw i8, ptr %1, i64 144
  store i32 36, ptr %70, align 8, !tbaa !6
  %71 = getelementptr inbounds nuw i8, ptr %1, i64 152
  store i32 38, ptr %71, align 8, !tbaa !6
  %72 = getelementptr inbounds nuw i8, ptr %1, i64 160
  store i32 40, ptr %72, align 8, !tbaa !6
  %73 = getelementptr inbounds nuw i8, ptr %1, i64 168
  store i32 42, ptr %73, align 8, !tbaa !6
  %74 = getelementptr inbounds nuw i8, ptr %1, i64 176
  store i32 44, ptr %74, align 8, !tbaa !6
  %75 = getelementptr inbounds nuw i8, ptr %1, i64 184
  store i32 46, ptr %75, align 8, !tbaa !6
  %76 = getelementptr inbounds nuw i8, ptr %1, i64 192
  store i32 48, ptr %76, align 8, !tbaa !6
  %77 = getelementptr inbounds nuw i8, ptr %1, i64 200
  store i32 50, ptr %77, align 8, !tbaa !6
  %78 = getelementptr inbounds nuw i8, ptr %1, i64 208
  store i32 52, ptr %78, align 8, !tbaa !6
  %79 = getelementptr inbounds nuw i8, ptr %1, i64 216
  store i32 54, ptr %79, align 8, !tbaa !6
  %80 = getelementptr inbounds nuw i8, ptr %1, i64 224
  store i32 56, ptr %80, align 8, !tbaa !6
  %81 = getelementptr inbounds nuw i8, ptr %1, i64 232
  store i32 58, ptr %81, align 8, !tbaa !6
  %82 = getelementptr inbounds nuw i8, ptr %1, i64 240
  store i32 60, ptr %82, align 8, !tbaa !6
  %83 = getelementptr inbounds nuw i8, ptr %1, i64 248
  store i32 62, ptr %83, align 8, !tbaa !6
  %84 = getelementptr inbounds nuw i8, ptr %1, i64 256
  store i32 64, ptr %84, align 8, !tbaa !6
  %85 = getelementptr inbounds nuw i8, ptr %1, i64 264
  store i32 66, ptr %85, align 8, !tbaa !6
  %86 = getelementptr inbounds nuw i8, ptr %1, i64 272
  store i32 68, ptr %86, align 8, !tbaa !6
  %87 = getelementptr inbounds nuw i8, ptr %1, i64 280
  store i32 70, ptr %87, align 8, !tbaa !6
  %88 = getelementptr inbounds nuw i8, ptr %1, i64 288
  store i32 72, ptr %88, align 8, !tbaa !6
  %89 = getelementptr inbounds nuw i8, ptr %1, i64 296
  store i32 74, ptr %89, align 8, !tbaa !6
  %90 = getelementptr inbounds nuw i8, ptr %1, i64 304
  store i32 76, ptr %90, align 8, !tbaa !6
  %91 = getelementptr inbounds nuw i8, ptr %1, i64 312
  store i32 78, ptr %91, align 8, !tbaa !6
  %92 = getelementptr inbounds nuw i8, ptr %1, i64 320
  store i32 80, ptr %92, align 8, !tbaa !6
  %93 = getelementptr inbounds nuw i8, ptr %1, i64 328
  store i32 82, ptr %93, align 8, !tbaa !6
  %94 = getelementptr inbounds nuw i8, ptr %1, i64 336
  store i32 84, ptr %94, align 8, !tbaa !6
  %95 = getelementptr inbounds nuw i8, ptr %1, i64 344
  store i32 86, ptr %95, align 8, !tbaa !6
  %96 = getelementptr inbounds nuw i8, ptr %1, i64 352
  store i32 88, ptr %96, align 8, !tbaa !6
  %97 = getelementptr inbounds nuw i8, ptr %1, i64 360
  store i32 90, ptr %97, align 8, !tbaa !6
  %98 = getelementptr inbounds nuw i8, ptr %1, i64 368
  store i32 92, ptr %98, align 8, !tbaa !6
  %99 = getelementptr inbounds nuw i8, ptr %1, i64 376
  store i32 94, ptr %99, align 8, !tbaa !6
  %100 = getelementptr inbounds nuw i8, ptr %1, i64 384
  store i32 96, ptr %100, align 8, !tbaa !6
  %101 = getelementptr inbounds nuw i8, ptr %1, i64 392
  store i32 98, ptr %101, align 8, !tbaa !6
  %102 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %103 = getelementptr inbounds nuw i8, ptr %1, i64 28
  store <4 x i32> <i32 7, i32 8, i32 9, i32 10>, ptr %102, align 4, !tbaa !6
  store <4 x i32> <i32 11, i32 12, i32 13, i32 14>, ptr %103, align 4, !tbaa !6
  %104 = getelementptr inbounds nuw i8, ptr %1, i64 44
  %105 = getelementptr inbounds nuw i8, ptr %1, i64 60
  store <4 x i32> <i32 15, i32 16, i32 17, i32 18>, ptr %104, align 4, !tbaa !6
  store <4 x i32> <i32 19, i32 20, i32 21, i32 22>, ptr %105, align 4, !tbaa !6
  %106 = getelementptr inbounds nuw i8, ptr %1, i64 76
  %107 = getelementptr inbounds nuw i8, ptr %1, i64 92
  store <4 x i32> <i32 23, i32 24, i32 25, i32 26>, ptr %106, align 4, !tbaa !6
  store <4 x i32> <i32 27, i32 28, i32 29, i32 30>, ptr %107, align 4, !tbaa !6
  %108 = getelementptr inbounds nuw i8, ptr %1, i64 108
  %109 = getelementptr inbounds nuw i8, ptr %1, i64 124
  store <4 x i32> <i32 31, i32 32, i32 33, i32 34>, ptr %108, align 4, !tbaa !6
  store <4 x i32> <i32 35, i32 36, i32 37, i32 38>, ptr %109, align 4, !tbaa !6
  %110 = getelementptr inbounds nuw i8, ptr %1, i64 140
  %111 = getelementptr inbounds nuw i8, ptr %1, i64 156
  store <4 x i32> <i32 39, i32 40, i32 41, i32 42>, ptr %110, align 4, !tbaa !6
  store <4 x i32> <i32 43, i32 44, i32 45, i32 46>, ptr %111, align 4, !tbaa !6
  %112 = getelementptr inbounds nuw i8, ptr %1, i64 172
  %113 = getelementptr inbounds nuw i8, ptr %1, i64 188
  store <4 x i32> <i32 47, i32 48, i32 49, i32 50>, ptr %112, align 4, !tbaa !6
  store <4 x i32> <i32 51, i32 52, i32 53, i32 54>, ptr %113, align 4, !tbaa !6
  %114 = getelementptr inbounds nuw i8, ptr %1, i64 204
  %115 = getelementptr inbounds nuw i8, ptr %1, i64 220
  store <4 x i32> <i32 55, i32 56, i32 57, i32 58>, ptr %114, align 4, !tbaa !6
  store <4 x i32> <i32 59, i32 60, i32 61, i32 62>, ptr %115, align 4, !tbaa !6
  %116 = getelementptr inbounds nuw i8, ptr %1, i64 236
  %117 = getelementptr inbounds nuw i8, ptr %1, i64 252
  store <4 x i32> <i32 63, i32 64, i32 65, i32 66>, ptr %116, align 4, !tbaa !6
  store <4 x i32> <i32 67, i32 68, i32 69, i32 70>, ptr %117, align 4, !tbaa !6
  %118 = getelementptr inbounds nuw i8, ptr %1, i64 268
  %119 = getelementptr inbounds nuw i8, ptr %1, i64 284
  store <4 x i32> <i32 71, i32 72, i32 73, i32 74>, ptr %118, align 4, !tbaa !6
  store <4 x i32> <i32 75, i32 76, i32 77, i32 78>, ptr %119, align 4, !tbaa !6
  %120 = getelementptr inbounds nuw i8, ptr %1, i64 300
  %121 = getelementptr inbounds nuw i8, ptr %1, i64 316
  store <4 x i32> <i32 79, i32 80, i32 81, i32 82>, ptr %120, align 4, !tbaa !6
  store <4 x i32> <i32 83, i32 84, i32 85, i32 86>, ptr %121, align 4, !tbaa !6
  %122 = getelementptr inbounds nuw i8, ptr %1, i64 332
  %123 = getelementptr inbounds nuw i8, ptr %1, i64 348
  store <4 x i32> <i32 87, i32 88, i32 89, i32 90>, ptr %122, align 4, !tbaa !6
  store <4 x i32> <i32 91, i32 92, i32 93, i32 94>, ptr %123, align 4, !tbaa !6
  %124 = getelementptr inbounds nuw i8, ptr %1, i64 364
  %125 = getelementptr inbounds nuw i8, ptr %1, i64 380
  store <4 x i32> <i32 95, i32 96, i32 97, i32 98>, ptr %124, align 4, !tbaa !6
  store <4 x i32> <i32 99, i32 100, i32 101, i32 102>, ptr %125, align 4, !tbaa !6
  %126 = getelementptr inbounds nuw i8, ptr %1, i64 396
  store i32 103, ptr %126, align 4, !tbaa !6
  %127 = getelementptr inbounds nuw i8, ptr %1, i64 400
  store i32 104, ptr %127, align 8, !tbaa !6
  %128 = getelementptr inbounds nuw i8, ptr %1, i64 404
  store i32 105, ptr %128, align 4, !tbaa !6
  %129 = getelementptr inbounds nuw i8, ptr %1, i64 408
  store i32 106, ptr %129, align 8, !tbaa !6
  br label %130

130:                                              ; preds = %0, %130
  %131 = phi i64 [ %184, %130 ], [ 13, %0 ]
  %132 = getelementptr inbounds nuw [200 x i32], ptr %1, i64 %131
  %133 = getelementptr inbounds nuw i32, ptr %132, i64 %131
  %134 = load i32, ptr %133, align 4, !tbaa !6
  store i32 %134, ptr %132, align 8, !tbaa !6
  %135 = getelementptr inbounds nuw i8, ptr %132, i64 4
  store i32 %134, ptr %135, align 4, !tbaa !6
  %136 = load i32, ptr %133, align 4, !tbaa !6
  %137 = getelementptr inbounds nuw i8, ptr %132, i64 8
  store i32 %136, ptr %137, align 8, !tbaa !6
  %138 = getelementptr inbounds nuw i8, ptr %132, i64 12
  store i32 %136, ptr %138, align 4, !tbaa !6
  %139 = load i32, ptr %133, align 4, !tbaa !6
  %140 = getelementptr inbounds nuw i8, ptr %132, i64 16
  store i32 %139, ptr %140, align 8, !tbaa !6
  %141 = getelementptr inbounds nuw i8, ptr %132, i64 20
  store i32 %139, ptr %141, align 4, !tbaa !6
  %142 = load i32, ptr %133, align 4, !tbaa !6
  %143 = getelementptr inbounds nuw i8, ptr %132, i64 24
  store i32 %142, ptr %143, align 8, !tbaa !6
  %144 = getelementptr inbounds nuw i8, ptr %132, i64 28
  store i32 %142, ptr %144, align 4, !tbaa !6
  %145 = load i32, ptr %133, align 4, !tbaa !6
  %146 = getelementptr inbounds nuw i8, ptr %132, i64 32
  store i32 %145, ptr %146, align 8, !tbaa !6
  %147 = getelementptr inbounds nuw i8, ptr %132, i64 36
  store i32 %145, ptr %147, align 4, !tbaa !6
  %148 = load i32, ptr %133, align 4, !tbaa !6
  %149 = getelementptr inbounds nuw i8, ptr %132, i64 40
  store i32 %148, ptr %149, align 8, !tbaa !6
  %150 = getelementptr inbounds nuw i8, ptr %132, i64 44
  store i32 %148, ptr %150, align 4, !tbaa !6
  %151 = load i32, ptr %133, align 4, !tbaa !6
  %152 = getelementptr inbounds nuw i8, ptr %132, i64 48
  store i32 %151, ptr %152, align 8, !tbaa !6
  %153 = getelementptr inbounds nuw i8, ptr %132, i64 52
  store i32 %151, ptr %153, align 4, !tbaa !6
  %154 = load i32, ptr %133, align 4, !tbaa !6
  %155 = getelementptr inbounds nuw i8, ptr %132, i64 56
  store i32 %154, ptr %155, align 8, !tbaa !6
  %156 = getelementptr inbounds nuw i8, ptr %132, i64 60
  store i32 %154, ptr %156, align 4, !tbaa !6
  %157 = load i32, ptr %133, align 4, !tbaa !6
  %158 = getelementptr inbounds nuw i8, ptr %132, i64 64
  store i32 %157, ptr %158, align 8, !tbaa !6
  %159 = getelementptr inbounds nuw i8, ptr %132, i64 68
  store i32 %157, ptr %159, align 4, !tbaa !6
  %160 = load i32, ptr %133, align 4, !tbaa !6
  %161 = getelementptr inbounds nuw i8, ptr %132, i64 72
  store i32 %160, ptr %161, align 8, !tbaa !6
  %162 = getelementptr inbounds nuw i8, ptr %132, i64 76
  store i32 %160, ptr %162, align 4, !tbaa !6
  %163 = load i32, ptr %133, align 4, !tbaa !6
  %164 = getelementptr inbounds nuw i8, ptr %132, i64 80
  store i32 %163, ptr %164, align 8, !tbaa !6
  %165 = getelementptr inbounds nuw i8, ptr %132, i64 84
  store i32 %163, ptr %165, align 4, !tbaa !6
  %166 = load i32, ptr %133, align 4, !tbaa !6
  %167 = getelementptr inbounds nuw i8, ptr %132, i64 88
  store i32 %166, ptr %167, align 8, !tbaa !6
  %168 = getelementptr inbounds nuw i8, ptr %132, i64 92
  store i32 %166, ptr %168, align 4, !tbaa !6
  %169 = load i32, ptr %133, align 4, !tbaa !6
  %170 = getelementptr inbounds nuw i8, ptr %132, i64 96
  store i32 %169, ptr %170, align 8, !tbaa !6
  %171 = getelementptr inbounds nuw i8, ptr %132, i64 100
  store i32 %169, ptr %171, align 4, !tbaa !6
  %172 = load i32, ptr %133, align 4, !tbaa !6
  %173 = getelementptr inbounds nuw i8, ptr %132, i64 104
  store i32 %172, ptr %173, align 8, !tbaa !6
  %174 = getelementptr inbounds nuw i8, ptr %132, i64 108
  store i32 %172, ptr %174, align 4, !tbaa !6
  %175 = load i32, ptr %133, align 4, !tbaa !6
  %176 = getelementptr inbounds nuw i8, ptr %132, i64 112
  store i32 %175, ptr %176, align 8, !tbaa !6
  %177 = getelementptr inbounds nuw i8, ptr %132, i64 116
  store i32 %175, ptr %177, align 4, !tbaa !6
  %178 = load i32, ptr %133, align 4, !tbaa !6
  %179 = getelementptr inbounds nuw i8, ptr %132, i64 120
  store i32 %178, ptr %179, align 8, !tbaa !6
  %180 = getelementptr inbounds nuw i8, ptr %132, i64 124
  store i32 %178, ptr %180, align 4, !tbaa !6
  %181 = load i32, ptr %133, align 4, !tbaa !6
  %182 = getelementptr inbounds nuw i8, ptr %132, i64 128
  store i32 %181, ptr %182, align 8, !tbaa !6
  %183 = getelementptr inbounds nuw i8, ptr %132, i64 132
  store i32 %181, ptr %183, align 4, !tbaa !6
  %184 = add nuw nsw i64 %131, 1
  %185 = icmp eq i64 %184, 100
  br i1 %185, label %186, label %130, !llvm.loop !10

186:                                              ; preds = %130, %203
  %187 = phi i64 [ %204, %203 ], [ 0, %130 ]
  %188 = phi double [ %200, %203 ], [ 0.000000e+00, %130 ]
  %189 = getelementptr inbounds nuw [200 x i32], ptr %1, i64 %187
  br label %190

190:                                              ; preds = %190, %186
  %191 = phi i64 [ 0, %186 ], [ %201, %190 ]
  %192 = phi double [ %188, %186 ], [ %200, %190 ]
  %193 = getelementptr inbounds nuw i32, ptr %189, i64 %191
  %194 = getelementptr inbounds nuw i8, ptr %193, i64 16
  %195 = load <4 x i32>, ptr %193, align 8, !tbaa !6
  %196 = load <4 x i32>, ptr %194, align 8, !tbaa !6
  %197 = sitofp <4 x i32> %195 to <4 x double>
  %198 = sitofp <4 x i32> %196 to <4 x double>
  %199 = tail call double @llvm.vector.reduce.fadd.v4f64(double %192, <4 x double> %197)
  %200 = tail call double @llvm.vector.reduce.fadd.v4f64(double %199, <4 x double> %198)
  %201 = add nuw i64 %191, 8
  %202 = icmp eq i64 %201, 200
  br i1 %202, label %203, label %190, !llvm.loop !12

203:                                              ; preds = %190
  %204 = add nuw nsw i64 %187, 2
  %205 = icmp samesign ult i64 %187, 98
  br i1 %205, label %186, label %206, !llvm.loop !15

206:                                              ; preds = %203
  %207 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %200)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.vector.reduce.fadd.v4f64(double, <4 x double>) #4

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }

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
!12 = distinct !{!12, !11, !13, !14}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
!15 = distinct !{!15, !11}
