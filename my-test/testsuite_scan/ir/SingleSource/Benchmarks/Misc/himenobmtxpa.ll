; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/himenobmtxpa.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/himenobmtxpa.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.Mat = type { ptr, i32, i32, i32, i32 }
%struct.timeval = type { i64, i64 }

@omega = dso_local local_unnamed_addr global float 0x3FE99999A0000000, align 4
@.str = private unnamed_addr constant [34 x i8] c"mimax = %d mjmax = %d mkmax = %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [30 x i8] c"imax = %d jmax = %d kmax =%d\0A\00", align 1
@p = dso_local global %struct.Mat zeroinitializer, align 8
@bnd = dso_local global %struct.Mat zeroinitializer, align 8
@wrk1 = dso_local global %struct.Mat zeroinitializer, align 8
@wrk2 = dso_local global %struct.Mat zeroinitializer, align 8
@a = dso_local global %struct.Mat zeroinitializer, align 8
@b = dso_local global %struct.Mat zeroinitializer, align 8
@c = dso_local global %struct.Mat zeroinitializer, align 8
@.str.2 = private unnamed_addr constant [29 x i8] c" Loop executed for %d times\0A\00", align 1
@.str.3 = private unnamed_addr constant [13 x i8] c" Gosa : %e \0A\00", align 1
@second.base_sec = internal unnamed_addr global i32 0, align 4
@second.base_usec = internal unnamed_addr global i32 0, align 4
@str = private unnamed_addr constant [27 x i8] c"Invalid input character !!\00", align 4

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 64, i32 noundef 64, i32 noundef 128)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 63, i32 noundef 63, i32 noundef 127)
  store <4 x i32> <i32 1, i32 64, i32 64, i32 128>, ptr getelementptr inbounds nuw (i8, ptr @p, i64 8), align 8, !tbaa !6
  %5 = tail call noalias dereferenceable_or_null(2097152) ptr @malloc(i64 noundef 2097152) #16
  store ptr %5, ptr @p, align 8, !tbaa !10
  store <4 x i32> <i32 1, i32 64, i32 64, i32 128>, ptr getelementptr inbounds nuw (i8, ptr @bnd, i64 8), align 8, !tbaa !6
  %6 = tail call noalias dereferenceable_or_null(2097152) ptr @malloc(i64 noundef 2097152) #16
  store ptr %6, ptr @bnd, align 8, !tbaa !10
  store <4 x i32> <i32 1, i32 64, i32 64, i32 128>, ptr getelementptr inbounds nuw (i8, ptr @wrk1, i64 8), align 8, !tbaa !6
  %7 = tail call noalias dereferenceable_or_null(2097152) ptr @malloc(i64 noundef 2097152) #16
  store ptr %7, ptr @wrk1, align 8, !tbaa !10
  store <4 x i32> <i32 1, i32 64, i32 64, i32 128>, ptr getelementptr inbounds nuw (i8, ptr @wrk2, i64 8), align 8, !tbaa !6
  %8 = tail call noalias dereferenceable_or_null(2097152) ptr @malloc(i64 noundef 2097152) #16
  store ptr %8, ptr @wrk2, align 8, !tbaa !10
  store <4 x i32> <i32 4, i32 64, i32 64, i32 128>, ptr getelementptr inbounds nuw (i8, ptr @a, i64 8), align 8, !tbaa !6
  %9 = tail call noalias dereferenceable_or_null(8388608) ptr @malloc(i64 noundef 8388608) #16
  store ptr %9, ptr @a, align 8, !tbaa !10
  store <4 x i32> <i32 3, i32 64, i32 64, i32 128>, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8, !tbaa !6
  %10 = tail call noalias dereferenceable_or_null(6291456) ptr @malloc(i64 noundef 6291456) #16
  store ptr %10, ptr @b, align 8, !tbaa !10
  store <4 x i32> <i32 3, i32 64, i32 64, i32 128>, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8, !tbaa !6
  %11 = tail call noalias dereferenceable_or_null(6291456) ptr @malloc(i64 noundef 6291456) #16
  store ptr %11, ptr @c, align 8, !tbaa !10
  br label %12

12:                                               ; preds = %61, %2
  %13 = phi i64 [ %62, %61 ], [ 0, %2 ]
  %14 = mul nuw nsw i64 %13, %13
  %15 = trunc nuw i64 %14 to i32
  %16 = uitofp nneg i32 %15 to float
  %17 = fdiv float %16, 3.969000e+03
  %18 = shl nuw nsw i64 %13, 6
  %19 = insertelement <4 x float> poison, float %17, i64 0
  %20 = shufflevector <4 x float> %19, <4 x float> poison, <4 x i32> zeroinitializer
  br label %21

21:                                               ; preds = %21, %12
  %22 = phi i64 [ %59, %21 ], [ 0, %12 ]
  %23 = add nuw nsw i64 %22, %18
  %24 = trunc nuw i64 %23 to i32
  %25 = shl i32 %24, 7
  %26 = sext i32 %25 to i64
  %27 = getelementptr float, ptr %5, i64 %26
  store <4 x float> %20, ptr %27, align 4, !tbaa !14
  %28 = getelementptr i8, ptr %27, i64 16
  store <4 x float> %20, ptr %28, align 4, !tbaa !14
  %29 = getelementptr i8, ptr %27, i64 32
  store <4 x float> %20, ptr %29, align 4, !tbaa !14
  %30 = getelementptr i8, ptr %27, i64 48
  store <4 x float> %20, ptr %30, align 4, !tbaa !14
  %31 = getelementptr i8, ptr %27, i64 64
  store <4 x float> %20, ptr %31, align 4, !tbaa !14
  %32 = getelementptr i8, ptr %27, i64 80
  store <4 x float> %20, ptr %32, align 4, !tbaa !14
  %33 = getelementptr i8, ptr %27, i64 96
  store <4 x float> %20, ptr %33, align 4, !tbaa !14
  %34 = getelementptr i8, ptr %27, i64 112
  store <4 x float> %20, ptr %34, align 4, !tbaa !14
  %35 = getelementptr i8, ptr %27, i64 128
  store <4 x float> %20, ptr %35, align 4, !tbaa !14
  %36 = getelementptr i8, ptr %27, i64 144
  store <4 x float> %20, ptr %36, align 4, !tbaa !14
  %37 = getelementptr i8, ptr %27, i64 160
  store <4 x float> %20, ptr %37, align 4, !tbaa !14
  %38 = getelementptr i8, ptr %27, i64 176
  store <4 x float> %20, ptr %38, align 4, !tbaa !14
  %39 = getelementptr i8, ptr %27, i64 192
  store <4 x float> %20, ptr %39, align 4, !tbaa !14
  %40 = getelementptr i8, ptr %27, i64 208
  store <4 x float> %20, ptr %40, align 4, !tbaa !14
  %41 = getelementptr i8, ptr %27, i64 224
  store <4 x float> %20, ptr %41, align 4, !tbaa !14
  %42 = getelementptr i8, ptr %27, i64 240
  store <4 x float> %20, ptr %42, align 4, !tbaa !14
  %43 = getelementptr i8, ptr %27, i64 256
  store <4 x float> %20, ptr %43, align 4, !tbaa !14
  %44 = getelementptr i8, ptr %27, i64 272
  store <4 x float> %20, ptr %44, align 4, !tbaa !14
  %45 = getelementptr i8, ptr %27, i64 288
  store <4 x float> %20, ptr %45, align 4, !tbaa !14
  %46 = getelementptr i8, ptr %27, i64 304
  store <4 x float> %20, ptr %46, align 4, !tbaa !14
  %47 = getelementptr i8, ptr %27, i64 320
  store <4 x float> %20, ptr %47, align 4, !tbaa !14
  %48 = getelementptr i8, ptr %27, i64 336
  store <4 x float> %20, ptr %48, align 4, !tbaa !14
  %49 = getelementptr i8, ptr %27, i64 352
  store <4 x float> %20, ptr %49, align 4, !tbaa !14
  %50 = getelementptr i8, ptr %27, i64 368
  store <4 x float> %20, ptr %50, align 4, !tbaa !14
  %51 = getelementptr i8, ptr %27, i64 384
  store <4 x float> %20, ptr %51, align 4, !tbaa !14
  %52 = getelementptr i8, ptr %27, i64 400
  store <4 x float> %20, ptr %52, align 4, !tbaa !14
  %53 = getelementptr i8, ptr %27, i64 416
  store <4 x float> %20, ptr %53, align 4, !tbaa !14
  %54 = getelementptr i8, ptr %27, i64 432
  store <4 x float> %20, ptr %54, align 4, !tbaa !14
  %55 = getelementptr i8, ptr %27, i64 448
  store <4 x float> %20, ptr %55, align 4, !tbaa !14
  %56 = getelementptr i8, ptr %27, i64 464
  store <4 x float> %20, ptr %56, align 4, !tbaa !14
  %57 = getelementptr i8, ptr %27, i64 480
  store <4 x float> %20, ptr %57, align 4, !tbaa !14
  %58 = getelementptr i8, ptr %27, i64 496
  store <4 x float> %20, ptr %58, align 4, !tbaa !14
  %59 = add nuw nsw i64 %22, 1
  %60 = icmp eq i64 %59, 64
  br i1 %60, label %61, label %21, !llvm.loop !16

61:                                               ; preds = %21
  %62 = add nuw nsw i64 %13, 1
  %63 = icmp eq i64 %62, 64
  br i1 %63, label %64, label %12, !llvm.loop !18

64:                                               ; preds = %61, %107
  %65 = phi i32 [ %108, %107 ], [ 0, %61 ]
  %66 = shl i32 %65, 13
  br label %67

67:                                               ; preds = %67, %64
  %68 = phi i64 [ %105, %67 ], [ 0, %64 ]
  %69 = trunc nuw nsw i64 %68 to i32
  %70 = shl i32 %69, 7
  %71 = add nuw nsw i32 %70, %66
  %72 = sext i32 %71 to i64
  %73 = getelementptr float, ptr %6, i64 %72
  store <4 x float> splat (float 1.000000e+00), ptr %73, align 4, !tbaa !14
  %74 = getelementptr i8, ptr %73, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %74, align 4, !tbaa !14
  %75 = getelementptr i8, ptr %73, i64 32
  store <4 x float> splat (float 1.000000e+00), ptr %75, align 4, !tbaa !14
  %76 = getelementptr i8, ptr %73, i64 48
  store <4 x float> splat (float 1.000000e+00), ptr %76, align 4, !tbaa !14
  %77 = getelementptr i8, ptr %73, i64 64
  store <4 x float> splat (float 1.000000e+00), ptr %77, align 4, !tbaa !14
  %78 = getelementptr i8, ptr %73, i64 80
  store <4 x float> splat (float 1.000000e+00), ptr %78, align 4, !tbaa !14
  %79 = getelementptr i8, ptr %73, i64 96
  store <4 x float> splat (float 1.000000e+00), ptr %79, align 4, !tbaa !14
  %80 = getelementptr i8, ptr %73, i64 112
  store <4 x float> splat (float 1.000000e+00), ptr %80, align 4, !tbaa !14
  %81 = getelementptr i8, ptr %73, i64 128
  store <4 x float> splat (float 1.000000e+00), ptr %81, align 4, !tbaa !14
  %82 = getelementptr i8, ptr %73, i64 144
  store <4 x float> splat (float 1.000000e+00), ptr %82, align 4, !tbaa !14
  %83 = getelementptr i8, ptr %73, i64 160
  store <4 x float> splat (float 1.000000e+00), ptr %83, align 4, !tbaa !14
  %84 = getelementptr i8, ptr %73, i64 176
  store <4 x float> splat (float 1.000000e+00), ptr %84, align 4, !tbaa !14
  %85 = getelementptr i8, ptr %73, i64 192
  store <4 x float> splat (float 1.000000e+00), ptr %85, align 4, !tbaa !14
  %86 = getelementptr i8, ptr %73, i64 208
  store <4 x float> splat (float 1.000000e+00), ptr %86, align 4, !tbaa !14
  %87 = getelementptr i8, ptr %73, i64 224
  store <4 x float> splat (float 1.000000e+00), ptr %87, align 4, !tbaa !14
  %88 = getelementptr i8, ptr %73, i64 240
  store <4 x float> splat (float 1.000000e+00), ptr %88, align 4, !tbaa !14
  %89 = getelementptr i8, ptr %73, i64 256
  store <4 x float> splat (float 1.000000e+00), ptr %89, align 4, !tbaa !14
  %90 = getelementptr i8, ptr %73, i64 272
  store <4 x float> splat (float 1.000000e+00), ptr %90, align 4, !tbaa !14
  %91 = getelementptr i8, ptr %73, i64 288
  store <4 x float> splat (float 1.000000e+00), ptr %91, align 4, !tbaa !14
  %92 = getelementptr i8, ptr %73, i64 304
  store <4 x float> splat (float 1.000000e+00), ptr %92, align 4, !tbaa !14
  %93 = getelementptr i8, ptr %73, i64 320
  store <4 x float> splat (float 1.000000e+00), ptr %93, align 4, !tbaa !14
  %94 = getelementptr i8, ptr %73, i64 336
  store <4 x float> splat (float 1.000000e+00), ptr %94, align 4, !tbaa !14
  %95 = getelementptr i8, ptr %73, i64 352
  store <4 x float> splat (float 1.000000e+00), ptr %95, align 4, !tbaa !14
  %96 = getelementptr i8, ptr %73, i64 368
  store <4 x float> splat (float 1.000000e+00), ptr %96, align 4, !tbaa !14
  %97 = getelementptr i8, ptr %73, i64 384
  store <4 x float> splat (float 1.000000e+00), ptr %97, align 4, !tbaa !14
  %98 = getelementptr i8, ptr %73, i64 400
  store <4 x float> splat (float 1.000000e+00), ptr %98, align 4, !tbaa !14
  %99 = getelementptr i8, ptr %73, i64 416
  store <4 x float> splat (float 1.000000e+00), ptr %99, align 4, !tbaa !14
  %100 = getelementptr i8, ptr %73, i64 432
  store <4 x float> splat (float 1.000000e+00), ptr %100, align 4, !tbaa !14
  %101 = getelementptr i8, ptr %73, i64 448
  store <4 x float> splat (float 1.000000e+00), ptr %101, align 4, !tbaa !14
  %102 = getelementptr i8, ptr %73, i64 464
  store <4 x float> splat (float 1.000000e+00), ptr %102, align 4, !tbaa !14
  %103 = getelementptr i8, ptr %73, i64 480
  store <4 x float> splat (float 1.000000e+00), ptr %103, align 4, !tbaa !14
  %104 = getelementptr i8, ptr %73, i64 496
  store <4 x float> splat (float 1.000000e+00), ptr %104, align 4, !tbaa !14
  %105 = add nuw nsw i64 %68, 1
  %106 = icmp eq i64 %105, 64
  br i1 %106, label %107, label %67, !llvm.loop !19

107:                                              ; preds = %67
  %108 = add nuw nsw i32 %65, 1
  %109 = icmp eq i32 %108, 64
  br i1 %109, label %110, label %64, !llvm.loop !20

110:                                              ; preds = %107
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(2097152) %7, i8 0, i64 2097152, i1 false), !tbaa !14
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(2097152) %8, i8 0, i64 2097152, i1 false), !tbaa !14
  br label %111

111:                                              ; preds = %110, %154
  %112 = phi i32 [ %155, %154 ], [ 0, %110 ]
  %113 = shl i32 %112, 13
  br label %114

114:                                              ; preds = %114, %111
  %115 = phi i64 [ %152, %114 ], [ 0, %111 ]
  %116 = trunc nuw nsw i64 %115 to i32
  %117 = shl i32 %116, 7
  %118 = add nuw nsw i32 %117, %113
  %119 = sext i32 %118 to i64
  %120 = getelementptr float, ptr %9, i64 %119
  store <4 x float> splat (float 1.000000e+00), ptr %120, align 4, !tbaa !14
  %121 = getelementptr i8, ptr %120, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %121, align 4, !tbaa !14
  %122 = getelementptr i8, ptr %120, i64 32
  store <4 x float> splat (float 1.000000e+00), ptr %122, align 4, !tbaa !14
  %123 = getelementptr i8, ptr %120, i64 48
  store <4 x float> splat (float 1.000000e+00), ptr %123, align 4, !tbaa !14
  %124 = getelementptr i8, ptr %120, i64 64
  store <4 x float> splat (float 1.000000e+00), ptr %124, align 4, !tbaa !14
  %125 = getelementptr i8, ptr %120, i64 80
  store <4 x float> splat (float 1.000000e+00), ptr %125, align 4, !tbaa !14
  %126 = getelementptr i8, ptr %120, i64 96
  store <4 x float> splat (float 1.000000e+00), ptr %126, align 4, !tbaa !14
  %127 = getelementptr i8, ptr %120, i64 112
  store <4 x float> splat (float 1.000000e+00), ptr %127, align 4, !tbaa !14
  %128 = getelementptr i8, ptr %120, i64 128
  store <4 x float> splat (float 1.000000e+00), ptr %128, align 4, !tbaa !14
  %129 = getelementptr i8, ptr %120, i64 144
  store <4 x float> splat (float 1.000000e+00), ptr %129, align 4, !tbaa !14
  %130 = getelementptr i8, ptr %120, i64 160
  store <4 x float> splat (float 1.000000e+00), ptr %130, align 4, !tbaa !14
  %131 = getelementptr i8, ptr %120, i64 176
  store <4 x float> splat (float 1.000000e+00), ptr %131, align 4, !tbaa !14
  %132 = getelementptr i8, ptr %120, i64 192
  store <4 x float> splat (float 1.000000e+00), ptr %132, align 4, !tbaa !14
  %133 = getelementptr i8, ptr %120, i64 208
  store <4 x float> splat (float 1.000000e+00), ptr %133, align 4, !tbaa !14
  %134 = getelementptr i8, ptr %120, i64 224
  store <4 x float> splat (float 1.000000e+00), ptr %134, align 4, !tbaa !14
  %135 = getelementptr i8, ptr %120, i64 240
  store <4 x float> splat (float 1.000000e+00), ptr %135, align 4, !tbaa !14
  %136 = getelementptr i8, ptr %120, i64 256
  store <4 x float> splat (float 1.000000e+00), ptr %136, align 4, !tbaa !14
  %137 = getelementptr i8, ptr %120, i64 272
  store <4 x float> splat (float 1.000000e+00), ptr %137, align 4, !tbaa !14
  %138 = getelementptr i8, ptr %120, i64 288
  store <4 x float> splat (float 1.000000e+00), ptr %138, align 4, !tbaa !14
  %139 = getelementptr i8, ptr %120, i64 304
  store <4 x float> splat (float 1.000000e+00), ptr %139, align 4, !tbaa !14
  %140 = getelementptr i8, ptr %120, i64 320
  store <4 x float> splat (float 1.000000e+00), ptr %140, align 4, !tbaa !14
  %141 = getelementptr i8, ptr %120, i64 336
  store <4 x float> splat (float 1.000000e+00), ptr %141, align 4, !tbaa !14
  %142 = getelementptr i8, ptr %120, i64 352
  store <4 x float> splat (float 1.000000e+00), ptr %142, align 4, !tbaa !14
  %143 = getelementptr i8, ptr %120, i64 368
  store <4 x float> splat (float 1.000000e+00), ptr %143, align 4, !tbaa !14
  %144 = getelementptr i8, ptr %120, i64 384
  store <4 x float> splat (float 1.000000e+00), ptr %144, align 4, !tbaa !14
  %145 = getelementptr i8, ptr %120, i64 400
  store <4 x float> splat (float 1.000000e+00), ptr %145, align 4, !tbaa !14
  %146 = getelementptr i8, ptr %120, i64 416
  store <4 x float> splat (float 1.000000e+00), ptr %146, align 4, !tbaa !14
  %147 = getelementptr i8, ptr %120, i64 432
  store <4 x float> splat (float 1.000000e+00), ptr %147, align 4, !tbaa !14
  %148 = getelementptr i8, ptr %120, i64 448
  store <4 x float> splat (float 1.000000e+00), ptr %148, align 4, !tbaa !14
  %149 = getelementptr i8, ptr %120, i64 464
  store <4 x float> splat (float 1.000000e+00), ptr %149, align 4, !tbaa !14
  %150 = getelementptr i8, ptr %120, i64 480
  store <4 x float> splat (float 1.000000e+00), ptr %150, align 4, !tbaa !14
  %151 = getelementptr i8, ptr %120, i64 496
  store <4 x float> splat (float 1.000000e+00), ptr %151, align 4, !tbaa !14
  %152 = add nuw nsw i64 %115, 1
  %153 = icmp eq i64 %152, 64
  br i1 %153, label %154, label %114, !llvm.loop !19

154:                                              ; preds = %114
  %155 = add nuw nsw i32 %112, 1
  %156 = icmp eq i32 %155, 64
  br i1 %156, label %157, label %111, !llvm.loop !20

157:                                              ; preds = %154, %201
  %158 = phi i32 [ %202, %201 ], [ 0, %154 ]
  %159 = shl i32 %158, 6
  %160 = add nuw nsw i32 %159, 4096
  br label %161

161:                                              ; preds = %161, %157
  %162 = phi i64 [ %199, %161 ], [ 0, %157 ]
  %163 = trunc nuw nsw i64 %162 to i32
  %164 = add nuw nsw i32 %160, %163
  %165 = shl i32 %164, 7
  %166 = sext i32 %165 to i64
  %167 = getelementptr float, ptr %9, i64 %166
  store <4 x float> splat (float 1.000000e+00), ptr %167, align 4, !tbaa !14
  %168 = getelementptr i8, ptr %167, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %168, align 4, !tbaa !14
  %169 = getelementptr i8, ptr %167, i64 32
  store <4 x float> splat (float 1.000000e+00), ptr %169, align 4, !tbaa !14
  %170 = getelementptr i8, ptr %167, i64 48
  store <4 x float> splat (float 1.000000e+00), ptr %170, align 4, !tbaa !14
  %171 = getelementptr i8, ptr %167, i64 64
  store <4 x float> splat (float 1.000000e+00), ptr %171, align 4, !tbaa !14
  %172 = getelementptr i8, ptr %167, i64 80
  store <4 x float> splat (float 1.000000e+00), ptr %172, align 4, !tbaa !14
  %173 = getelementptr i8, ptr %167, i64 96
  store <4 x float> splat (float 1.000000e+00), ptr %173, align 4, !tbaa !14
  %174 = getelementptr i8, ptr %167, i64 112
  store <4 x float> splat (float 1.000000e+00), ptr %174, align 4, !tbaa !14
  %175 = getelementptr i8, ptr %167, i64 128
  store <4 x float> splat (float 1.000000e+00), ptr %175, align 4, !tbaa !14
  %176 = getelementptr i8, ptr %167, i64 144
  store <4 x float> splat (float 1.000000e+00), ptr %176, align 4, !tbaa !14
  %177 = getelementptr i8, ptr %167, i64 160
  store <4 x float> splat (float 1.000000e+00), ptr %177, align 4, !tbaa !14
  %178 = getelementptr i8, ptr %167, i64 176
  store <4 x float> splat (float 1.000000e+00), ptr %178, align 4, !tbaa !14
  %179 = getelementptr i8, ptr %167, i64 192
  store <4 x float> splat (float 1.000000e+00), ptr %179, align 4, !tbaa !14
  %180 = getelementptr i8, ptr %167, i64 208
  store <4 x float> splat (float 1.000000e+00), ptr %180, align 4, !tbaa !14
  %181 = getelementptr i8, ptr %167, i64 224
  store <4 x float> splat (float 1.000000e+00), ptr %181, align 4, !tbaa !14
  %182 = getelementptr i8, ptr %167, i64 240
  store <4 x float> splat (float 1.000000e+00), ptr %182, align 4, !tbaa !14
  %183 = getelementptr i8, ptr %167, i64 256
  store <4 x float> splat (float 1.000000e+00), ptr %183, align 4, !tbaa !14
  %184 = getelementptr i8, ptr %167, i64 272
  store <4 x float> splat (float 1.000000e+00), ptr %184, align 4, !tbaa !14
  %185 = getelementptr i8, ptr %167, i64 288
  store <4 x float> splat (float 1.000000e+00), ptr %185, align 4, !tbaa !14
  %186 = getelementptr i8, ptr %167, i64 304
  store <4 x float> splat (float 1.000000e+00), ptr %186, align 4, !tbaa !14
  %187 = getelementptr i8, ptr %167, i64 320
  store <4 x float> splat (float 1.000000e+00), ptr %187, align 4, !tbaa !14
  %188 = getelementptr i8, ptr %167, i64 336
  store <4 x float> splat (float 1.000000e+00), ptr %188, align 4, !tbaa !14
  %189 = getelementptr i8, ptr %167, i64 352
  store <4 x float> splat (float 1.000000e+00), ptr %189, align 4, !tbaa !14
  %190 = getelementptr i8, ptr %167, i64 368
  store <4 x float> splat (float 1.000000e+00), ptr %190, align 4, !tbaa !14
  %191 = getelementptr i8, ptr %167, i64 384
  store <4 x float> splat (float 1.000000e+00), ptr %191, align 4, !tbaa !14
  %192 = getelementptr i8, ptr %167, i64 400
  store <4 x float> splat (float 1.000000e+00), ptr %192, align 4, !tbaa !14
  %193 = getelementptr i8, ptr %167, i64 416
  store <4 x float> splat (float 1.000000e+00), ptr %193, align 4, !tbaa !14
  %194 = getelementptr i8, ptr %167, i64 432
  store <4 x float> splat (float 1.000000e+00), ptr %194, align 4, !tbaa !14
  %195 = getelementptr i8, ptr %167, i64 448
  store <4 x float> splat (float 1.000000e+00), ptr %195, align 4, !tbaa !14
  %196 = getelementptr i8, ptr %167, i64 464
  store <4 x float> splat (float 1.000000e+00), ptr %196, align 4, !tbaa !14
  %197 = getelementptr i8, ptr %167, i64 480
  store <4 x float> splat (float 1.000000e+00), ptr %197, align 4, !tbaa !14
  %198 = getelementptr i8, ptr %167, i64 496
  store <4 x float> splat (float 1.000000e+00), ptr %198, align 4, !tbaa !14
  %199 = add nuw nsw i64 %162, 1
  %200 = icmp eq i64 %199, 64
  br i1 %200, label %201, label %161, !llvm.loop !19

201:                                              ; preds = %161
  %202 = add nuw nsw i32 %158, 1
  %203 = icmp eq i32 %202, 64
  br i1 %203, label %204, label %157, !llvm.loop !20

204:                                              ; preds = %201, %248
  %205 = phi i32 [ %249, %248 ], [ 0, %201 ]
  %206 = shl i32 %205, 6
  %207 = add nuw nsw i32 %206, 8192
  br label %208

208:                                              ; preds = %208, %204
  %209 = phi i64 [ %246, %208 ], [ 0, %204 ]
  %210 = trunc nuw nsw i64 %209 to i32
  %211 = add nuw nsw i32 %207, %210
  %212 = shl i32 %211, 7
  %213 = sext i32 %212 to i64
  %214 = getelementptr float, ptr %9, i64 %213
  store <4 x float> splat (float 1.000000e+00), ptr %214, align 4, !tbaa !14
  %215 = getelementptr i8, ptr %214, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %215, align 4, !tbaa !14
  %216 = getelementptr i8, ptr %214, i64 32
  store <4 x float> splat (float 1.000000e+00), ptr %216, align 4, !tbaa !14
  %217 = getelementptr i8, ptr %214, i64 48
  store <4 x float> splat (float 1.000000e+00), ptr %217, align 4, !tbaa !14
  %218 = getelementptr i8, ptr %214, i64 64
  store <4 x float> splat (float 1.000000e+00), ptr %218, align 4, !tbaa !14
  %219 = getelementptr i8, ptr %214, i64 80
  store <4 x float> splat (float 1.000000e+00), ptr %219, align 4, !tbaa !14
  %220 = getelementptr i8, ptr %214, i64 96
  store <4 x float> splat (float 1.000000e+00), ptr %220, align 4, !tbaa !14
  %221 = getelementptr i8, ptr %214, i64 112
  store <4 x float> splat (float 1.000000e+00), ptr %221, align 4, !tbaa !14
  %222 = getelementptr i8, ptr %214, i64 128
  store <4 x float> splat (float 1.000000e+00), ptr %222, align 4, !tbaa !14
  %223 = getelementptr i8, ptr %214, i64 144
  store <4 x float> splat (float 1.000000e+00), ptr %223, align 4, !tbaa !14
  %224 = getelementptr i8, ptr %214, i64 160
  store <4 x float> splat (float 1.000000e+00), ptr %224, align 4, !tbaa !14
  %225 = getelementptr i8, ptr %214, i64 176
  store <4 x float> splat (float 1.000000e+00), ptr %225, align 4, !tbaa !14
  %226 = getelementptr i8, ptr %214, i64 192
  store <4 x float> splat (float 1.000000e+00), ptr %226, align 4, !tbaa !14
  %227 = getelementptr i8, ptr %214, i64 208
  store <4 x float> splat (float 1.000000e+00), ptr %227, align 4, !tbaa !14
  %228 = getelementptr i8, ptr %214, i64 224
  store <4 x float> splat (float 1.000000e+00), ptr %228, align 4, !tbaa !14
  %229 = getelementptr i8, ptr %214, i64 240
  store <4 x float> splat (float 1.000000e+00), ptr %229, align 4, !tbaa !14
  %230 = getelementptr i8, ptr %214, i64 256
  store <4 x float> splat (float 1.000000e+00), ptr %230, align 4, !tbaa !14
  %231 = getelementptr i8, ptr %214, i64 272
  store <4 x float> splat (float 1.000000e+00), ptr %231, align 4, !tbaa !14
  %232 = getelementptr i8, ptr %214, i64 288
  store <4 x float> splat (float 1.000000e+00), ptr %232, align 4, !tbaa !14
  %233 = getelementptr i8, ptr %214, i64 304
  store <4 x float> splat (float 1.000000e+00), ptr %233, align 4, !tbaa !14
  %234 = getelementptr i8, ptr %214, i64 320
  store <4 x float> splat (float 1.000000e+00), ptr %234, align 4, !tbaa !14
  %235 = getelementptr i8, ptr %214, i64 336
  store <4 x float> splat (float 1.000000e+00), ptr %235, align 4, !tbaa !14
  %236 = getelementptr i8, ptr %214, i64 352
  store <4 x float> splat (float 1.000000e+00), ptr %236, align 4, !tbaa !14
  %237 = getelementptr i8, ptr %214, i64 368
  store <4 x float> splat (float 1.000000e+00), ptr %237, align 4, !tbaa !14
  %238 = getelementptr i8, ptr %214, i64 384
  store <4 x float> splat (float 1.000000e+00), ptr %238, align 4, !tbaa !14
  %239 = getelementptr i8, ptr %214, i64 400
  store <4 x float> splat (float 1.000000e+00), ptr %239, align 4, !tbaa !14
  %240 = getelementptr i8, ptr %214, i64 416
  store <4 x float> splat (float 1.000000e+00), ptr %240, align 4, !tbaa !14
  %241 = getelementptr i8, ptr %214, i64 432
  store <4 x float> splat (float 1.000000e+00), ptr %241, align 4, !tbaa !14
  %242 = getelementptr i8, ptr %214, i64 448
  store <4 x float> splat (float 1.000000e+00), ptr %242, align 4, !tbaa !14
  %243 = getelementptr i8, ptr %214, i64 464
  store <4 x float> splat (float 1.000000e+00), ptr %243, align 4, !tbaa !14
  %244 = getelementptr i8, ptr %214, i64 480
  store <4 x float> splat (float 1.000000e+00), ptr %244, align 4, !tbaa !14
  %245 = getelementptr i8, ptr %214, i64 496
  store <4 x float> splat (float 1.000000e+00), ptr %245, align 4, !tbaa !14
  %246 = add nuw nsw i64 %209, 1
  %247 = icmp eq i64 %246, 64
  br i1 %247, label %248, label %208, !llvm.loop !19

248:                                              ; preds = %208
  %249 = add nuw nsw i32 %205, 1
  %250 = icmp eq i32 %249, 64
  br i1 %250, label %251, label %204, !llvm.loop !20

251:                                              ; preds = %248, %295
  %252 = phi i32 [ %296, %295 ], [ 0, %248 ]
  %253 = shl i32 %252, 6
  %254 = add nuw nsw i32 %253, 12288
  br label %255

255:                                              ; preds = %255, %251
  %256 = phi i64 [ %293, %255 ], [ 0, %251 ]
  %257 = trunc nuw nsw i64 %256 to i32
  %258 = add nuw nsw i32 %254, %257
  %259 = shl i32 %258, 7
  %260 = sext i32 %259 to i64
  %261 = getelementptr float, ptr %9, i64 %260
  store <4 x float> splat (float 0x3FC5555560000000), ptr %261, align 4, !tbaa !14
  %262 = getelementptr i8, ptr %261, i64 16
  store <4 x float> splat (float 0x3FC5555560000000), ptr %262, align 4, !tbaa !14
  %263 = getelementptr i8, ptr %261, i64 32
  store <4 x float> splat (float 0x3FC5555560000000), ptr %263, align 4, !tbaa !14
  %264 = getelementptr i8, ptr %261, i64 48
  store <4 x float> splat (float 0x3FC5555560000000), ptr %264, align 4, !tbaa !14
  %265 = getelementptr i8, ptr %261, i64 64
  store <4 x float> splat (float 0x3FC5555560000000), ptr %265, align 4, !tbaa !14
  %266 = getelementptr i8, ptr %261, i64 80
  store <4 x float> splat (float 0x3FC5555560000000), ptr %266, align 4, !tbaa !14
  %267 = getelementptr i8, ptr %261, i64 96
  store <4 x float> splat (float 0x3FC5555560000000), ptr %267, align 4, !tbaa !14
  %268 = getelementptr i8, ptr %261, i64 112
  store <4 x float> splat (float 0x3FC5555560000000), ptr %268, align 4, !tbaa !14
  %269 = getelementptr i8, ptr %261, i64 128
  store <4 x float> splat (float 0x3FC5555560000000), ptr %269, align 4, !tbaa !14
  %270 = getelementptr i8, ptr %261, i64 144
  store <4 x float> splat (float 0x3FC5555560000000), ptr %270, align 4, !tbaa !14
  %271 = getelementptr i8, ptr %261, i64 160
  store <4 x float> splat (float 0x3FC5555560000000), ptr %271, align 4, !tbaa !14
  %272 = getelementptr i8, ptr %261, i64 176
  store <4 x float> splat (float 0x3FC5555560000000), ptr %272, align 4, !tbaa !14
  %273 = getelementptr i8, ptr %261, i64 192
  store <4 x float> splat (float 0x3FC5555560000000), ptr %273, align 4, !tbaa !14
  %274 = getelementptr i8, ptr %261, i64 208
  store <4 x float> splat (float 0x3FC5555560000000), ptr %274, align 4, !tbaa !14
  %275 = getelementptr i8, ptr %261, i64 224
  store <4 x float> splat (float 0x3FC5555560000000), ptr %275, align 4, !tbaa !14
  %276 = getelementptr i8, ptr %261, i64 240
  store <4 x float> splat (float 0x3FC5555560000000), ptr %276, align 4, !tbaa !14
  %277 = getelementptr i8, ptr %261, i64 256
  store <4 x float> splat (float 0x3FC5555560000000), ptr %277, align 4, !tbaa !14
  %278 = getelementptr i8, ptr %261, i64 272
  store <4 x float> splat (float 0x3FC5555560000000), ptr %278, align 4, !tbaa !14
  %279 = getelementptr i8, ptr %261, i64 288
  store <4 x float> splat (float 0x3FC5555560000000), ptr %279, align 4, !tbaa !14
  %280 = getelementptr i8, ptr %261, i64 304
  store <4 x float> splat (float 0x3FC5555560000000), ptr %280, align 4, !tbaa !14
  %281 = getelementptr i8, ptr %261, i64 320
  store <4 x float> splat (float 0x3FC5555560000000), ptr %281, align 4, !tbaa !14
  %282 = getelementptr i8, ptr %261, i64 336
  store <4 x float> splat (float 0x3FC5555560000000), ptr %282, align 4, !tbaa !14
  %283 = getelementptr i8, ptr %261, i64 352
  store <4 x float> splat (float 0x3FC5555560000000), ptr %283, align 4, !tbaa !14
  %284 = getelementptr i8, ptr %261, i64 368
  store <4 x float> splat (float 0x3FC5555560000000), ptr %284, align 4, !tbaa !14
  %285 = getelementptr i8, ptr %261, i64 384
  store <4 x float> splat (float 0x3FC5555560000000), ptr %285, align 4, !tbaa !14
  %286 = getelementptr i8, ptr %261, i64 400
  store <4 x float> splat (float 0x3FC5555560000000), ptr %286, align 4, !tbaa !14
  %287 = getelementptr i8, ptr %261, i64 416
  store <4 x float> splat (float 0x3FC5555560000000), ptr %287, align 4, !tbaa !14
  %288 = getelementptr i8, ptr %261, i64 432
  store <4 x float> splat (float 0x3FC5555560000000), ptr %288, align 4, !tbaa !14
  %289 = getelementptr i8, ptr %261, i64 448
  store <4 x float> splat (float 0x3FC5555560000000), ptr %289, align 4, !tbaa !14
  %290 = getelementptr i8, ptr %261, i64 464
  store <4 x float> splat (float 0x3FC5555560000000), ptr %290, align 4, !tbaa !14
  %291 = getelementptr i8, ptr %261, i64 480
  store <4 x float> splat (float 0x3FC5555560000000), ptr %291, align 4, !tbaa !14
  %292 = getelementptr i8, ptr %261, i64 496
  store <4 x float> splat (float 0x3FC5555560000000), ptr %292, align 4, !tbaa !14
  %293 = add nuw nsw i64 %256, 1
  %294 = icmp eq i64 %293, 64
  br i1 %294, label %295, label %255, !llvm.loop !19

295:                                              ; preds = %255
  %296 = add nuw nsw i32 %252, 1
  %297 = icmp eq i32 %296, 64
  br i1 %297, label %298, label %251, !llvm.loop !20

298:                                              ; preds = %295
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(6291456) %10, i8 0, i64 6291456, i1 false)
  br label %299

299:                                              ; preds = %298, %342
  %300 = phi i32 [ %343, %342 ], [ 0, %298 ]
  %301 = shl i32 %300, 13
  br label %302

302:                                              ; preds = %302, %299
  %303 = phi i64 [ %340, %302 ], [ 0, %299 ]
  %304 = trunc nuw nsw i64 %303 to i32
  %305 = shl i32 %304, 7
  %306 = add nuw nsw i32 %305, %301
  %307 = sext i32 %306 to i64
  %308 = getelementptr float, ptr %11, i64 %307
  store <4 x float> splat (float 1.000000e+00), ptr %308, align 4, !tbaa !14
  %309 = getelementptr i8, ptr %308, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %309, align 4, !tbaa !14
  %310 = getelementptr i8, ptr %308, i64 32
  store <4 x float> splat (float 1.000000e+00), ptr %310, align 4, !tbaa !14
  %311 = getelementptr i8, ptr %308, i64 48
  store <4 x float> splat (float 1.000000e+00), ptr %311, align 4, !tbaa !14
  %312 = getelementptr i8, ptr %308, i64 64
  store <4 x float> splat (float 1.000000e+00), ptr %312, align 4, !tbaa !14
  %313 = getelementptr i8, ptr %308, i64 80
  store <4 x float> splat (float 1.000000e+00), ptr %313, align 4, !tbaa !14
  %314 = getelementptr i8, ptr %308, i64 96
  store <4 x float> splat (float 1.000000e+00), ptr %314, align 4, !tbaa !14
  %315 = getelementptr i8, ptr %308, i64 112
  store <4 x float> splat (float 1.000000e+00), ptr %315, align 4, !tbaa !14
  %316 = getelementptr i8, ptr %308, i64 128
  store <4 x float> splat (float 1.000000e+00), ptr %316, align 4, !tbaa !14
  %317 = getelementptr i8, ptr %308, i64 144
  store <4 x float> splat (float 1.000000e+00), ptr %317, align 4, !tbaa !14
  %318 = getelementptr i8, ptr %308, i64 160
  store <4 x float> splat (float 1.000000e+00), ptr %318, align 4, !tbaa !14
  %319 = getelementptr i8, ptr %308, i64 176
  store <4 x float> splat (float 1.000000e+00), ptr %319, align 4, !tbaa !14
  %320 = getelementptr i8, ptr %308, i64 192
  store <4 x float> splat (float 1.000000e+00), ptr %320, align 4, !tbaa !14
  %321 = getelementptr i8, ptr %308, i64 208
  store <4 x float> splat (float 1.000000e+00), ptr %321, align 4, !tbaa !14
  %322 = getelementptr i8, ptr %308, i64 224
  store <4 x float> splat (float 1.000000e+00), ptr %322, align 4, !tbaa !14
  %323 = getelementptr i8, ptr %308, i64 240
  store <4 x float> splat (float 1.000000e+00), ptr %323, align 4, !tbaa !14
  %324 = getelementptr i8, ptr %308, i64 256
  store <4 x float> splat (float 1.000000e+00), ptr %324, align 4, !tbaa !14
  %325 = getelementptr i8, ptr %308, i64 272
  store <4 x float> splat (float 1.000000e+00), ptr %325, align 4, !tbaa !14
  %326 = getelementptr i8, ptr %308, i64 288
  store <4 x float> splat (float 1.000000e+00), ptr %326, align 4, !tbaa !14
  %327 = getelementptr i8, ptr %308, i64 304
  store <4 x float> splat (float 1.000000e+00), ptr %327, align 4, !tbaa !14
  %328 = getelementptr i8, ptr %308, i64 320
  store <4 x float> splat (float 1.000000e+00), ptr %328, align 4, !tbaa !14
  %329 = getelementptr i8, ptr %308, i64 336
  store <4 x float> splat (float 1.000000e+00), ptr %329, align 4, !tbaa !14
  %330 = getelementptr i8, ptr %308, i64 352
  store <4 x float> splat (float 1.000000e+00), ptr %330, align 4, !tbaa !14
  %331 = getelementptr i8, ptr %308, i64 368
  store <4 x float> splat (float 1.000000e+00), ptr %331, align 4, !tbaa !14
  %332 = getelementptr i8, ptr %308, i64 384
  store <4 x float> splat (float 1.000000e+00), ptr %332, align 4, !tbaa !14
  %333 = getelementptr i8, ptr %308, i64 400
  store <4 x float> splat (float 1.000000e+00), ptr %333, align 4, !tbaa !14
  %334 = getelementptr i8, ptr %308, i64 416
  store <4 x float> splat (float 1.000000e+00), ptr %334, align 4, !tbaa !14
  %335 = getelementptr i8, ptr %308, i64 432
  store <4 x float> splat (float 1.000000e+00), ptr %335, align 4, !tbaa !14
  %336 = getelementptr i8, ptr %308, i64 448
  store <4 x float> splat (float 1.000000e+00), ptr %336, align 4, !tbaa !14
  %337 = getelementptr i8, ptr %308, i64 464
  store <4 x float> splat (float 1.000000e+00), ptr %337, align 4, !tbaa !14
  %338 = getelementptr i8, ptr %308, i64 480
  store <4 x float> splat (float 1.000000e+00), ptr %338, align 4, !tbaa !14
  %339 = getelementptr i8, ptr %308, i64 496
  store <4 x float> splat (float 1.000000e+00), ptr %339, align 4, !tbaa !14
  %340 = add nuw nsw i64 %303, 1
  %341 = icmp eq i64 %340, 64
  br i1 %341, label %342, label %302, !llvm.loop !19

342:                                              ; preds = %302
  %343 = add nuw nsw i32 %300, 1
  %344 = icmp eq i32 %343, 64
  br i1 %344, label %345, label %299, !llvm.loop !20

345:                                              ; preds = %342, %389
  %346 = phi i32 [ %390, %389 ], [ 0, %342 ]
  %347 = shl i32 %346, 6
  %348 = add nuw nsw i32 %347, 4096
  br label %349

349:                                              ; preds = %349, %345
  %350 = phi i64 [ %387, %349 ], [ 0, %345 ]
  %351 = trunc nuw nsw i64 %350 to i32
  %352 = add nuw nsw i32 %348, %351
  %353 = shl i32 %352, 7
  %354 = sext i32 %353 to i64
  %355 = getelementptr float, ptr %11, i64 %354
  store <4 x float> splat (float 1.000000e+00), ptr %355, align 4, !tbaa !14
  %356 = getelementptr i8, ptr %355, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %356, align 4, !tbaa !14
  %357 = getelementptr i8, ptr %355, i64 32
  store <4 x float> splat (float 1.000000e+00), ptr %357, align 4, !tbaa !14
  %358 = getelementptr i8, ptr %355, i64 48
  store <4 x float> splat (float 1.000000e+00), ptr %358, align 4, !tbaa !14
  %359 = getelementptr i8, ptr %355, i64 64
  store <4 x float> splat (float 1.000000e+00), ptr %359, align 4, !tbaa !14
  %360 = getelementptr i8, ptr %355, i64 80
  store <4 x float> splat (float 1.000000e+00), ptr %360, align 4, !tbaa !14
  %361 = getelementptr i8, ptr %355, i64 96
  store <4 x float> splat (float 1.000000e+00), ptr %361, align 4, !tbaa !14
  %362 = getelementptr i8, ptr %355, i64 112
  store <4 x float> splat (float 1.000000e+00), ptr %362, align 4, !tbaa !14
  %363 = getelementptr i8, ptr %355, i64 128
  store <4 x float> splat (float 1.000000e+00), ptr %363, align 4, !tbaa !14
  %364 = getelementptr i8, ptr %355, i64 144
  store <4 x float> splat (float 1.000000e+00), ptr %364, align 4, !tbaa !14
  %365 = getelementptr i8, ptr %355, i64 160
  store <4 x float> splat (float 1.000000e+00), ptr %365, align 4, !tbaa !14
  %366 = getelementptr i8, ptr %355, i64 176
  store <4 x float> splat (float 1.000000e+00), ptr %366, align 4, !tbaa !14
  %367 = getelementptr i8, ptr %355, i64 192
  store <4 x float> splat (float 1.000000e+00), ptr %367, align 4, !tbaa !14
  %368 = getelementptr i8, ptr %355, i64 208
  store <4 x float> splat (float 1.000000e+00), ptr %368, align 4, !tbaa !14
  %369 = getelementptr i8, ptr %355, i64 224
  store <4 x float> splat (float 1.000000e+00), ptr %369, align 4, !tbaa !14
  %370 = getelementptr i8, ptr %355, i64 240
  store <4 x float> splat (float 1.000000e+00), ptr %370, align 4, !tbaa !14
  %371 = getelementptr i8, ptr %355, i64 256
  store <4 x float> splat (float 1.000000e+00), ptr %371, align 4, !tbaa !14
  %372 = getelementptr i8, ptr %355, i64 272
  store <4 x float> splat (float 1.000000e+00), ptr %372, align 4, !tbaa !14
  %373 = getelementptr i8, ptr %355, i64 288
  store <4 x float> splat (float 1.000000e+00), ptr %373, align 4, !tbaa !14
  %374 = getelementptr i8, ptr %355, i64 304
  store <4 x float> splat (float 1.000000e+00), ptr %374, align 4, !tbaa !14
  %375 = getelementptr i8, ptr %355, i64 320
  store <4 x float> splat (float 1.000000e+00), ptr %375, align 4, !tbaa !14
  %376 = getelementptr i8, ptr %355, i64 336
  store <4 x float> splat (float 1.000000e+00), ptr %376, align 4, !tbaa !14
  %377 = getelementptr i8, ptr %355, i64 352
  store <4 x float> splat (float 1.000000e+00), ptr %377, align 4, !tbaa !14
  %378 = getelementptr i8, ptr %355, i64 368
  store <4 x float> splat (float 1.000000e+00), ptr %378, align 4, !tbaa !14
  %379 = getelementptr i8, ptr %355, i64 384
  store <4 x float> splat (float 1.000000e+00), ptr %379, align 4, !tbaa !14
  %380 = getelementptr i8, ptr %355, i64 400
  store <4 x float> splat (float 1.000000e+00), ptr %380, align 4, !tbaa !14
  %381 = getelementptr i8, ptr %355, i64 416
  store <4 x float> splat (float 1.000000e+00), ptr %381, align 4, !tbaa !14
  %382 = getelementptr i8, ptr %355, i64 432
  store <4 x float> splat (float 1.000000e+00), ptr %382, align 4, !tbaa !14
  %383 = getelementptr i8, ptr %355, i64 448
  store <4 x float> splat (float 1.000000e+00), ptr %383, align 4, !tbaa !14
  %384 = getelementptr i8, ptr %355, i64 464
  store <4 x float> splat (float 1.000000e+00), ptr %384, align 4, !tbaa !14
  %385 = getelementptr i8, ptr %355, i64 480
  store <4 x float> splat (float 1.000000e+00), ptr %385, align 4, !tbaa !14
  %386 = getelementptr i8, ptr %355, i64 496
  store <4 x float> splat (float 1.000000e+00), ptr %386, align 4, !tbaa !14
  %387 = add nuw nsw i64 %350, 1
  %388 = icmp eq i64 %387, 64
  br i1 %388, label %389, label %349, !llvm.loop !19

389:                                              ; preds = %349
  %390 = add nuw nsw i32 %346, 1
  %391 = icmp eq i32 %390, 64
  br i1 %391, label %392, label %345, !llvm.loop !20

392:                                              ; preds = %389, %436
  %393 = phi i32 [ %437, %436 ], [ 0, %389 ]
  %394 = shl i32 %393, 6
  %395 = add nuw nsw i32 %394, 8192
  br label %396

396:                                              ; preds = %396, %392
  %397 = phi i64 [ %434, %396 ], [ 0, %392 ]
  %398 = trunc nuw nsw i64 %397 to i32
  %399 = add nuw nsw i32 %395, %398
  %400 = shl i32 %399, 7
  %401 = sext i32 %400 to i64
  %402 = getelementptr float, ptr %11, i64 %401
  store <4 x float> splat (float 1.000000e+00), ptr %402, align 4, !tbaa !14
  %403 = getelementptr i8, ptr %402, i64 16
  store <4 x float> splat (float 1.000000e+00), ptr %403, align 4, !tbaa !14
  %404 = getelementptr i8, ptr %402, i64 32
  store <4 x float> splat (float 1.000000e+00), ptr %404, align 4, !tbaa !14
  %405 = getelementptr i8, ptr %402, i64 48
  store <4 x float> splat (float 1.000000e+00), ptr %405, align 4, !tbaa !14
  %406 = getelementptr i8, ptr %402, i64 64
  store <4 x float> splat (float 1.000000e+00), ptr %406, align 4, !tbaa !14
  %407 = getelementptr i8, ptr %402, i64 80
  store <4 x float> splat (float 1.000000e+00), ptr %407, align 4, !tbaa !14
  %408 = getelementptr i8, ptr %402, i64 96
  store <4 x float> splat (float 1.000000e+00), ptr %408, align 4, !tbaa !14
  %409 = getelementptr i8, ptr %402, i64 112
  store <4 x float> splat (float 1.000000e+00), ptr %409, align 4, !tbaa !14
  %410 = getelementptr i8, ptr %402, i64 128
  store <4 x float> splat (float 1.000000e+00), ptr %410, align 4, !tbaa !14
  %411 = getelementptr i8, ptr %402, i64 144
  store <4 x float> splat (float 1.000000e+00), ptr %411, align 4, !tbaa !14
  %412 = getelementptr i8, ptr %402, i64 160
  store <4 x float> splat (float 1.000000e+00), ptr %412, align 4, !tbaa !14
  %413 = getelementptr i8, ptr %402, i64 176
  store <4 x float> splat (float 1.000000e+00), ptr %413, align 4, !tbaa !14
  %414 = getelementptr i8, ptr %402, i64 192
  store <4 x float> splat (float 1.000000e+00), ptr %414, align 4, !tbaa !14
  %415 = getelementptr i8, ptr %402, i64 208
  store <4 x float> splat (float 1.000000e+00), ptr %415, align 4, !tbaa !14
  %416 = getelementptr i8, ptr %402, i64 224
  store <4 x float> splat (float 1.000000e+00), ptr %416, align 4, !tbaa !14
  %417 = getelementptr i8, ptr %402, i64 240
  store <4 x float> splat (float 1.000000e+00), ptr %417, align 4, !tbaa !14
  %418 = getelementptr i8, ptr %402, i64 256
  store <4 x float> splat (float 1.000000e+00), ptr %418, align 4, !tbaa !14
  %419 = getelementptr i8, ptr %402, i64 272
  store <4 x float> splat (float 1.000000e+00), ptr %419, align 4, !tbaa !14
  %420 = getelementptr i8, ptr %402, i64 288
  store <4 x float> splat (float 1.000000e+00), ptr %420, align 4, !tbaa !14
  %421 = getelementptr i8, ptr %402, i64 304
  store <4 x float> splat (float 1.000000e+00), ptr %421, align 4, !tbaa !14
  %422 = getelementptr i8, ptr %402, i64 320
  store <4 x float> splat (float 1.000000e+00), ptr %422, align 4, !tbaa !14
  %423 = getelementptr i8, ptr %402, i64 336
  store <4 x float> splat (float 1.000000e+00), ptr %423, align 4, !tbaa !14
  %424 = getelementptr i8, ptr %402, i64 352
  store <4 x float> splat (float 1.000000e+00), ptr %424, align 4, !tbaa !14
  %425 = getelementptr i8, ptr %402, i64 368
  store <4 x float> splat (float 1.000000e+00), ptr %425, align 4, !tbaa !14
  %426 = getelementptr i8, ptr %402, i64 384
  store <4 x float> splat (float 1.000000e+00), ptr %426, align 4, !tbaa !14
  %427 = getelementptr i8, ptr %402, i64 400
  store <4 x float> splat (float 1.000000e+00), ptr %427, align 4, !tbaa !14
  %428 = getelementptr i8, ptr %402, i64 416
  store <4 x float> splat (float 1.000000e+00), ptr %428, align 4, !tbaa !14
  %429 = getelementptr i8, ptr %402, i64 432
  store <4 x float> splat (float 1.000000e+00), ptr %429, align 4, !tbaa !14
  %430 = getelementptr i8, ptr %402, i64 448
  store <4 x float> splat (float 1.000000e+00), ptr %430, align 4, !tbaa !14
  %431 = getelementptr i8, ptr %402, i64 464
  store <4 x float> splat (float 1.000000e+00), ptr %431, align 4, !tbaa !14
  %432 = getelementptr i8, ptr %402, i64 480
  store <4 x float> splat (float 1.000000e+00), ptr %432, align 4, !tbaa !14
  %433 = getelementptr i8, ptr %402, i64 496
  store <4 x float> splat (float 1.000000e+00), ptr %433, align 4, !tbaa !14
  %434 = add nuw nsw i64 %397, 1
  %435 = icmp eq i64 %434, 64
  br i1 %435, label %436, label %396, !llvm.loop !19

436:                                              ; preds = %396
  %437 = add nuw nsw i32 %393, 1
  %438 = icmp eq i32 %437, 64
  br i1 %438, label %439, label %392, !llvm.loop !20

439:                                              ; preds = %436
  %440 = tail call float @jacobi(i32 noundef 64, ptr noundef nonnull @a, ptr noundef nonnull @b, ptr noundef nonnull @c, ptr noundef nonnull @p, ptr noundef nonnull @bnd, ptr noundef nonnull @wrk1, ptr noundef nonnull @wrk2)
  %441 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 64)
  %442 = fpext float %440 to double
  %443 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, double noundef %442)
  %444 = load ptr, ptr @p, align 8, !tbaa !10
  %445 = icmp eq ptr %444, null
  br i1 %445, label %447, label %446

446:                                              ; preds = %439
  tail call void @free(ptr noundef nonnull %444) #17
  br label %447

447:                                              ; preds = %439, %446
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @p, i8 0, i64 24, i1 false)
  %448 = load ptr, ptr @bnd, align 8, !tbaa !10
  %449 = icmp eq ptr %448, null
  br i1 %449, label %451, label %450

450:                                              ; preds = %447
  tail call void @free(ptr noundef nonnull %448) #17
  br label %451

451:                                              ; preds = %447, %450
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @bnd, i8 0, i64 24, i1 false)
  %452 = load ptr, ptr @wrk1, align 8, !tbaa !10
  %453 = icmp eq ptr %452, null
  br i1 %453, label %455, label %454

454:                                              ; preds = %451
  tail call void @free(ptr noundef nonnull %452) #17
  br label %455

455:                                              ; preds = %451, %454
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @wrk1, i8 0, i64 24, i1 false)
  %456 = load ptr, ptr @wrk2, align 8, !tbaa !10
  %457 = icmp eq ptr %456, null
  br i1 %457, label %459, label %458

458:                                              ; preds = %455
  tail call void @free(ptr noundef nonnull %456) #17
  br label %459

459:                                              ; preds = %455, %458
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @wrk2, i8 0, i64 24, i1 false)
  %460 = load ptr, ptr @a, align 8, !tbaa !10
  %461 = icmp eq ptr %460, null
  br i1 %461, label %463, label %462

462:                                              ; preds = %459
  tail call void @free(ptr noundef nonnull %460) #17
  br label %463

463:                                              ; preds = %459, %462
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @a, i8 0, i64 24, i1 false)
  %464 = load ptr, ptr @b, align 8, !tbaa !10
  %465 = icmp eq ptr %464, null
  br i1 %465, label %467, label %466

466:                                              ; preds = %463
  tail call void @free(ptr noundef nonnull %464) #17
  br label %467

467:                                              ; preds = %463, %466
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @b, i8 0, i64 24, i1 false)
  %468 = load ptr, ptr @c, align 8, !tbaa !10
  %469 = icmp eq ptr %468, null
  br i1 %469, label %471, label %470

470:                                              ; preds = %467
  tail call void @free(ptr noundef nonnull %468) #17
  br label %471

471:                                              ; preds = %467, %470
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @c, i8 0, i64 24, i1 false)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: write, inaccessiblemem: readwrite) uwtable
define dso_local range(i32 0, 2) i32 @newMat(ptr noundef writeonly captures(none) initializes((0, 24)) %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #3 {
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i32 %1, ptr %6, align 8, !tbaa !21
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 12
  store i32 %2, ptr %7, align 4, !tbaa !22
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i32 %3, ptr %8, align 8, !tbaa !23
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 20
  store i32 %4, ptr %9, align 4, !tbaa !24
  %10 = mul nsw i32 %2, %1
  %11 = mul nsw i32 %10, %3
  %12 = mul nsw i32 %11, %4
  %13 = sext i32 %12 to i64
  %14 = shl nsw i64 %13, 2
  %15 = tail call noalias ptr @malloc(i64 noundef %14) #16
  store ptr %15, ptr %0, align 8, !tbaa !10
  %16 = icmp ne ptr %15, null
  %17 = zext i1 %16 to i32
  ret i32 %17
}

; Function Attrs: nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @mat_set_init(ptr noundef readonly captures(none) %0) local_unnamed_addr #4 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %3 = load i32, ptr %2, align 4, !tbaa !22
  %4 = icmp sgt i32 %3, 0
  br i1 %4, label %5, label %60

5:                                                ; preds = %1
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %7 = load i32, ptr %6, align 8, !tbaa !23
  %8 = icmp sgt i32 %7, 0
  %9 = add nsw i32 %3, -1
  %10 = mul nsw i32 %9, %9
  %11 = uitofp nneg i32 %10 to float
  br i1 %8, label %12, label %60

12:                                               ; preds = %5
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %14 = load i32, ptr %13, align 4, !tbaa !24
  %15 = icmp sgt i32 %14, 0
  br i1 %15, label %16, label %60

16:                                               ; preds = %12
  %17 = load ptr, ptr %0, align 8, !tbaa !10
  %18 = zext nneg i32 %7 to i64
  %19 = zext nneg i32 %3 to i64
  %20 = zext nneg i32 %14 to i64
  %21 = icmp ult i32 %14, 8
  %22 = and i64 %20, 2147483640
  %23 = icmp eq i64 %22, %20
  br label %24

24:                                               ; preds = %57, %16
  %25 = phi i64 [ %58, %57 ], [ 0, %16 ]
  %26 = mul nuw nsw i64 %25, %25
  %27 = trunc nuw i64 %26 to i32
  %28 = uitofp nneg i32 %27 to float
  %29 = fdiv float %28, %11
  %30 = mul nuw nsw i64 %25, %18
  %31 = insertelement <4 x float> poison, float %29, i64 0
  %32 = shufflevector <4 x float> %31, <4 x float> poison, <4 x i32> zeroinitializer
  br label %33

33:                                               ; preds = %54, %24
  %34 = phi i64 [ %55, %54 ], [ 0, %24 ]
  %35 = add nuw nsw i64 %30, %34
  %36 = trunc nuw i64 %35 to i32
  %37 = mul i32 %14, %36
  %38 = sext i32 %37 to i64
  %39 = getelementptr float, ptr %17, i64 %38
  br i1 %21, label %47, label %40

40:                                               ; preds = %33, %40
  %41 = phi i64 [ %44, %40 ], [ 0, %33 ]
  %42 = getelementptr float, ptr %39, i64 %41
  %43 = getelementptr i8, ptr %42, i64 16
  store <4 x float> %32, ptr %42, align 4, !tbaa !14
  store <4 x float> %32, ptr %43, align 4, !tbaa !14
  %44 = add nuw i64 %41, 8
  %45 = icmp eq i64 %44, %22
  br i1 %45, label %46, label %40, !llvm.loop !25

46:                                               ; preds = %40
  br i1 %23, label %54, label %47

47:                                               ; preds = %33, %46
  %48 = phi i64 [ 0, %33 ], [ %22, %46 ]
  br label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %52, %49 ], [ %48, %47 ]
  %51 = getelementptr float, ptr %39, i64 %50
  store float %29, ptr %51, align 4, !tbaa !14
  %52 = add nuw nsw i64 %50, 1
  %53 = icmp eq i64 %52, %20
  br i1 %53, label %54, label %49, !llvm.loop !28

54:                                               ; preds = %49, %46
  %55 = add nuw nsw i64 %34, 1
  %56 = icmp eq i64 %55, %18
  br i1 %56, label %57, label %33, !llvm.loop !16

57:                                               ; preds = %54
  %58 = add nuw nsw i64 %25, 1
  %59 = icmp eq i64 %58, %19
  br i1 %59, label %60, label %24, !llvm.loop !18

60:                                               ; preds = %57, %12, %5, %1
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @mat_set(ptr noundef readonly captures(none) %0, i32 noundef %1, float noundef %2) local_unnamed_addr #4 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %5 = load i32, ptr %4, align 4, !tbaa !22
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %56

7:                                                ; preds = %3
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = load i32, ptr %8, align 8, !tbaa !23
  %10 = icmp sgt i32 %9, 0
  %11 = mul nsw i32 %5, %1
  br i1 %10, label %12, label %56

12:                                               ; preds = %7
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %14 = load i32, ptr %13, align 4, !tbaa !24
  %15 = icmp sgt i32 %14, 0
  br i1 %15, label %16, label %56

16:                                               ; preds = %12
  %17 = load ptr, ptr %0, align 8, !tbaa !10
  %18 = zext nneg i32 %9 to i64
  %19 = zext nneg i32 %14 to i64
  %20 = icmp ult i32 %14, 8
  %21 = and i64 %19, 2147483640
  %22 = insertelement <4 x float> poison, float %2, i64 0
  %23 = shufflevector <4 x float> %22, <4 x float> poison, <4 x i32> zeroinitializer
  %24 = icmp eq i64 %21, %19
  br label %25

25:                                               ; preds = %53, %16
  %26 = phi i32 [ 0, %16 ], [ %54, %53 ]
  %27 = add i32 %11, %26
  %28 = mul i32 %9, %27
  br label %29

29:                                               ; preds = %50, %25
  %30 = phi i64 [ %51, %50 ], [ 0, %25 ]
  %31 = trunc nuw nsw i64 %30 to i32
  %32 = add i32 %28, %31
  %33 = mul i32 %14, %32
  %34 = sext i32 %33 to i64
  %35 = getelementptr float, ptr %17, i64 %34
  br i1 %20, label %43, label %36

36:                                               ; preds = %29, %36
  %37 = phi i64 [ %40, %36 ], [ 0, %29 ]
  %38 = getelementptr float, ptr %35, i64 %37
  %39 = getelementptr i8, ptr %38, i64 16
  store <4 x float> %23, ptr %38, align 4, !tbaa !14
  store <4 x float> %23, ptr %39, align 4, !tbaa !14
  %40 = add nuw i64 %37, 8
  %41 = icmp eq i64 %40, %21
  br i1 %41, label %42, label %36, !llvm.loop !29

42:                                               ; preds = %36
  br i1 %24, label %50, label %43

43:                                               ; preds = %29, %42
  %44 = phi i64 [ 0, %29 ], [ %21, %42 ]
  br label %45

45:                                               ; preds = %43, %45
  %46 = phi i64 [ %48, %45 ], [ %44, %43 ]
  %47 = getelementptr float, ptr %35, i64 %46
  store float %2, ptr %47, align 4, !tbaa !14
  %48 = add nuw nsw i64 %46, 1
  %49 = icmp eq i64 %48, %19
  br i1 %49, label %50, label %45, !llvm.loop !30

50:                                               ; preds = %45, %42
  %51 = add nuw nsw i64 %30, 1
  %52 = icmp eq i64 %51, %18
  br i1 %52, label %53, label %29, !llvm.loop !19

53:                                               ; preds = %50
  %54 = add nuw nsw i32 %26, 1
  %55 = icmp eq i32 %54, %5
  br i1 %55, label %56, label %25, !llvm.loop !20

56:                                               ; preds = %53, %12, %7, %3
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local float @jacobi(i32 noundef %0, ptr noundef readonly captures(none) %1, ptr noundef readonly captures(none) %2, ptr noundef readonly captures(none) %3, ptr noundef readonly captures(none) %4, ptr noundef readonly captures(none) %5, ptr noundef readonly captures(none) %6, ptr noundef readonly captures(none) %7) local_unnamed_addr #5 {
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 20
  %10 = load i32, ptr %9, align 4, !tbaa !24
  %11 = icmp sgt i32 %0, 0
  br i1 %11, label %12, label %909

12:                                               ; preds = %8
  %13 = add i32 %10, -1
  %14 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %15 = load i32, ptr %14, align 8, !tbaa !23
  %16 = add i32 %15, -1
  %17 = getelementptr inbounds nuw i8, ptr %4, i64 12
  %18 = load i32, ptr %17, align 4, !tbaa !22
  %19 = add i32 %18, -1
  %20 = icmp slt i32 %18, 3
  %21 = mul i32 %10, %15
  %22 = icmp slt i32 %15, 3
  %23 = icmp slt i32 %10, 3
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %27 = getelementptr inbounds nuw i8, ptr %2, i64 12
  %28 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %29 = getelementptr inbounds nuw i8, ptr %2, i64 20
  %30 = getelementptr inbounds nuw i8, ptr %3, i64 12
  %31 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %32 = getelementptr inbounds nuw i8, ptr %3, i64 20
  %33 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %34 = getelementptr inbounds nuw i8, ptr %6, i64 20
  %35 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %36 = getelementptr inbounds nuw i8, ptr %5, i64 20
  %37 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %38 = getelementptr inbounds nuw i8, ptr %7, i64 20
  %39 = sext i32 %10 to i64
  %40 = zext i32 %15 to i64
  %41 = zext nneg i32 %19 to i64
  %42 = zext i32 %16 to i64
  %43 = zext i32 %13 to i64
  %44 = zext nneg i32 %19 to i64
  %45 = zext i32 %16 to i64
  %46 = zext i32 %13 to i64
  %47 = add i32 %15, 1
  %48 = mul i32 %10, %47
  %49 = mul i32 %15, %10
  %50 = add nsw i64 %43, -1
  %51 = shl nuw nsw i64 %43, 2
  %52 = mul i32 %10, %15
  %53 = mul i32 %10, %15
  %54 = add nuw nsw i64 %42, 4611686018427387902
  %55 = mul i64 %54, %39
  %56 = shl i64 %55, 2
  %57 = add i64 %56, %51
  %58 = add i64 %57, 4
  %59 = shl nsw i64 %39, 3
  %60 = mul nsw i64 %39, %42
  %61 = add i64 %60, %43
  %62 = shl i64 %61, 2
  %63 = add i64 %56, %51
  %64 = or disjoint i64 %59, 4
  %65 = add nsw i64 %43, -1
  %66 = select i1 %20, i1 true, i1 %22
  %67 = select i1 %66, i1 true, i1 %23
  %68 = icmp ult i64 %65, 8
  %69 = and i64 %65, -8
  %70 = or disjoint i64 %69, 1
  %71 = icmp eq i64 %65, %69
  %72 = select i1 %20, i1 true, i1 %22
  %73 = select i1 %72, i1 true, i1 %23
  %74 = icmp ult i64 %50, 8
  %75 = and i64 %50, -8
  %76 = or disjoint i64 %75, 1
  %77 = icmp eq i64 %50, %75
  br label %78

78:                                               ; preds = %12, %905
  %79 = phi i32 [ 0, %12 ], [ %907, %905 ]
  br i1 %67, label %905, label %80

80:                                               ; preds = %78
  %81 = load ptr, ptr %1, align 8, !tbaa !10
  %82 = load i32, ptr %24, align 4, !tbaa !22
  %83 = load i32, ptr %25, align 8, !tbaa !23
  %84 = load i32, ptr %26, align 4, !tbaa !24
  %85 = load ptr, ptr %4, align 8, !tbaa !10
  %86 = shl i32 %82, 1
  %87 = load ptr, ptr %2, align 8, !tbaa !10
  %88 = load i32, ptr %27, align 4, !tbaa !22
  %89 = load i32, ptr %28, align 8, !tbaa !23
  %90 = load i32, ptr %29, align 4, !tbaa !24
  %91 = shl i32 %88, 1
  %92 = load ptr, ptr %3, align 8, !tbaa !10
  %93 = load i32, ptr %30, align 4, !tbaa !22
  %94 = load i32, ptr %31, align 8, !tbaa !23
  %95 = load i32, ptr %32, align 4, !tbaa !24
  %96 = shl i32 %93, 1
  %97 = load ptr, ptr %6, align 8, !tbaa !10
  %98 = load i32, ptr %33, align 8, !tbaa !23
  %99 = load i32, ptr %34, align 4, !tbaa !24
  %100 = mul i32 %82, 3
  %101 = load ptr, ptr %5, align 8, !tbaa !10
  %102 = load i32, ptr %35, align 8, !tbaa !23
  %103 = load i32, ptr %36, align 4, !tbaa !24
  %104 = load ptr, ptr %7, align 8, !tbaa !10
  %105 = load i32, ptr %37, align 8, !tbaa !23
  %106 = load i32, ptr %38, align 4, !tbaa !24
  %107 = getelementptr i8, ptr %104, i64 4
  %108 = add i32 %105, 1
  %109 = mul i32 %106, %108
  %110 = mul i32 %105, %106
  %111 = getelementptr i8, ptr %104, i64 %51
  %112 = getelementptr i8, ptr %81, i64 4
  %113 = add i32 %100, 1
  %114 = mul i32 %83, %113
  %115 = add i32 %114, 1
  %116 = mul i32 %84, %115
  %117 = mul i32 %83, %84
  %118 = getelementptr i8, ptr %81, i64 %51
  %119 = getelementptr i8, ptr %81, i64 4
  %120 = or disjoint i32 %86, 1
  %121 = mul i32 %83, %120
  %122 = add i32 %121, 1
  %123 = mul i32 %84, %122
  %124 = getelementptr i8, ptr %81, i64 %51
  %125 = getelementptr i8, ptr %81, i64 4
  %126 = add i32 %82, 1
  %127 = mul i32 %83, %126
  %128 = add i32 %127, 1
  %129 = mul i32 %84, %128
  %130 = getelementptr i8, ptr %81, i64 %51
  %131 = getelementptr i8, ptr %81, i64 4
  %132 = add i32 %83, 1
  %133 = mul i32 %84, %132
  %134 = getelementptr i8, ptr %81, i64 %51
  %135 = getelementptr i8, ptr %85, i64 %58
  %136 = getelementptr i8, ptr %85, i64 %58
  %137 = getelementptr i8, ptr %85, i64 %58
  %138 = getelementptr i8, ptr %85, i64 %58
  %139 = getelementptr i8, ptr %85, i64 %59
  %140 = getelementptr i8, ptr %85, i64 %62
  %141 = getelementptr i8, ptr %140, i64 4
  %142 = getelementptr i8, ptr %85, i64 4
  %143 = getelementptr i8, ptr %85, i64 %63
  %144 = getelementptr i8, ptr %85, i64 %64
  %145 = getelementptr i8, ptr %85, i64 %62
  %146 = getelementptr i8, ptr %85, i64 4
  %147 = getelementptr i8, ptr %85, i64 %63
  %148 = getelementptr i8, ptr %85, i64 %64
  %149 = getelementptr i8, ptr %85, i64 %62
  %150 = getelementptr i8, ptr %87, i64 4
  %151 = or disjoint i32 %91, 1
  %152 = mul i32 %89, %151
  %153 = add i32 %152, 1
  %154 = mul i32 %90, %153
  %155 = mul i32 %89, %90
  %156 = getelementptr i8, ptr %87, i64 %51
  %157 = getelementptr i8, ptr %87, i64 4
  %158 = add i32 %88, 1
  %159 = mul i32 %89, %158
  %160 = add i32 %159, 1
  %161 = mul i32 %90, %160
  %162 = getelementptr i8, ptr %87, i64 %51
  %163 = getelementptr i8, ptr %87, i64 4
  %164 = add i32 %89, 1
  %165 = mul i32 %90, %164
  %166 = getelementptr i8, ptr %87, i64 %51
  %167 = getelementptr i8, ptr %92, i64 4
  %168 = or disjoint i32 %96, 1
  %169 = mul i32 %94, %168
  %170 = add i32 %169, 1
  %171 = mul i32 %95, %170
  %172 = mul i32 %94, %95
  %173 = getelementptr i8, ptr %92, i64 %51
  %174 = getelementptr i8, ptr %92, i64 4
  %175 = add i32 %93, 1
  %176 = mul i32 %94, %175
  %177 = add i32 %176, 1
  %178 = mul i32 %95, %177
  %179 = getelementptr i8, ptr %92, i64 %51
  %180 = getelementptr i8, ptr %92, i64 4
  %181 = add i32 %94, 1
  %182 = mul i32 %95, %181
  %183 = getelementptr i8, ptr %92, i64 %51
  %184 = getelementptr i8, ptr %97, i64 4
  %185 = add i32 %98, 1
  %186 = mul i32 %99, %185
  %187 = mul i32 %98, %99
  %188 = getelementptr i8, ptr %97, i64 %51
  %189 = getelementptr i8, ptr %101, i64 4
  %190 = add i32 %102, 1
  %191 = mul i32 %103, %190
  %192 = mul i32 %102, %103
  %193 = getelementptr i8, ptr %101, i64 %51
  br label %194

194:                                              ; preds = %826, %80
  %195 = phi i32 [ %828, %826 ], [ 0, %80 ]
  %196 = phi i64 [ %253, %826 ], [ 1, %80 ]
  %197 = phi float [ %823, %826 ], [ 0.000000e+00, %80 ]
  %198 = mul i32 %110, %195
  %199 = add i32 %109, %198
  %200 = mul i32 %117, %195
  %201 = add i32 %116, %200
  %202 = add i32 %123, %200
  %203 = add i32 %129, %200
  %204 = add i32 %133, %200
  %205 = mul i32 %53, %195
  %206 = add i32 %52, %205
  %207 = sext i32 %206 to i64
  %208 = add nsw i64 %39, %207
  %209 = shl nsw i64 %208, 2
  %210 = getelementptr i8, ptr %85, i64 %209
  %211 = getelementptr i8, ptr %135, i64 %209
  %212 = shl nsw i64 %207, 2
  %213 = getelementptr i8, ptr %85, i64 %212
  %214 = getelementptr i8, ptr %136, i64 %212
  %215 = sext i32 %205 to i64
  %216 = add nsw i64 %39, %215
  %217 = shl nsw i64 %216, 2
  %218 = getelementptr i8, ptr %85, i64 %217
  %219 = getelementptr i8, ptr %137, i64 %217
  %220 = add i32 %195, 2
  %221 = mul i32 %53, %220
  %222 = sext i32 %221 to i64
  %223 = add nsw i64 %39, %222
  %224 = shl nsw i64 %223, 2
  %225 = getelementptr i8, ptr %85, i64 %224
  %226 = getelementptr i8, ptr %138, i64 %224
  %227 = getelementptr i8, ptr %139, i64 %212
  %228 = getelementptr i8, ptr %141, i64 %212
  %229 = shl nsw i64 %215, 2
  %230 = getelementptr i8, ptr %142, i64 %229
  %231 = getelementptr i8, ptr %143, i64 %229
  %232 = getelementptr i8, ptr %144, i64 %229
  %233 = getelementptr i8, ptr %145, i64 %229
  %234 = shl nsw i64 %222, 2
  %235 = getelementptr i8, ptr %146, i64 %234
  %236 = getelementptr i8, ptr %147, i64 %234
  %237 = getelementptr i8, ptr %148, i64 %234
  %238 = getelementptr i8, ptr %149, i64 %234
  %239 = mul i32 %155, %195
  %240 = add i32 %154, %239
  %241 = add i32 %161, %239
  %242 = add i32 %165, %239
  %243 = mul i32 %172, %195
  %244 = add i32 %171, %243
  %245 = add i32 %178, %243
  %246 = add i32 %182, %243
  %247 = mul i32 %187, %195
  %248 = add i32 %186, %247
  %249 = mul i32 %192, %195
  %250 = add i32 %191, %249
  %251 = trunc nuw nsw i64 %196 to i32
  %252 = mul i32 %21, %251
  %253 = add nuw nsw i64 %196, 1
  %254 = trunc nuw nsw i64 %253 to i32
  %255 = mul i32 %21, %254
  %256 = trunc i64 %196 to i32
  %257 = add i32 %256, -1
  %258 = mul i32 %21, %257
  %259 = add i32 %82, %251
  %260 = mul i32 %259, %83
  %261 = add i32 %86, %251
  %262 = mul i32 %261, %83
  %263 = add i32 %88, %251
  %264 = mul i32 %263, %89
  %265 = add i32 %91, %251
  %266 = mul i32 %265, %89
  %267 = add i32 %93, %251
  %268 = mul i32 %267, %94
  %269 = add i32 %96, %251
  %270 = mul i32 %269, %94
  %271 = add i32 %100, %251
  %272 = mul i32 %271, %83
  %273 = sext i32 %255 to i64
  %274 = sext i32 %252 to i64
  %275 = sext i32 %258 to i64
  %276 = trunc i64 %196 to i32
  %277 = mul i32 %83, %276
  %278 = trunc i64 %196 to i32
  %279 = mul i32 %89, %278
  %280 = trunc i64 %196 to i32
  %281 = mul i32 %94, %280
  %282 = trunc i64 %196 to i32
  %283 = mul i32 %98, %282
  %284 = trunc i64 %196 to i32
  %285 = mul i32 %102, %284
  %286 = trunc i64 %196 to i32
  %287 = mul i32 %105, %286
  %288 = getelementptr float, ptr %85, i64 %273
  %289 = getelementptr float, ptr %85, i64 %273
  %290 = getelementptr float, ptr %85, i64 %275
  %291 = getelementptr float, ptr %85, i64 %275
  br label %292

292:                                              ; preds = %822, %194
  %293 = phi i32 [ %825, %822 ], [ 0, %194 ]
  %294 = phi i64 [ %303, %822 ], [ 1, %194 ]
  %295 = phi float [ %823, %822 ], [ %197, %194 ]
  %296 = trunc nuw nsw i64 %294 to i32
  %297 = add i32 %277, %296
  %298 = mul i32 %297, %84
  %299 = mul nuw nsw i64 %294, %39
  %300 = add nsw i64 %299, %273
  %301 = add i32 %260, %296
  %302 = mul i32 %301, %84
  %303 = add nuw nsw i64 %294, 1
  %304 = mul nuw nsw i64 %303, %39
  %305 = add nsw i64 %304, %274
  %306 = add i32 %262, %296
  %307 = mul i32 %306, %84
  %308 = add nsw i64 %299, %274
  %309 = add i32 %279, %296
  %310 = mul i32 %309, %90
  %311 = add nsw i64 %294, -1
  %312 = mul nsw i64 %311, %39
  %313 = add i32 %264, %296
  %314 = mul i32 %313, %90
  %315 = add nsw i64 %312, %274
  %316 = add i32 %266, %296
  %317 = mul i32 %316, %90
  %318 = add nsw i64 %299, %275
  %319 = add i32 %281, %296
  %320 = mul i32 %319, %95
  %321 = add i32 %268, %296
  %322 = mul i32 %321, %95
  %323 = add i32 %270, %296
  %324 = mul i32 %323, %95
  %325 = add i32 %283, %296
  %326 = mul i32 %325, %99
  %327 = add i32 %272, %296
  %328 = mul i32 %327, %84
  %329 = add i32 %285, %296
  %330 = mul i32 %329, %103
  %331 = add i32 %287, %296
  %332 = mul i32 %331, %106
  %333 = sext i32 %298 to i64
  %334 = sext i32 %302 to i64
  %335 = sext i32 %307 to i64
  %336 = sext i32 %310 to i64
  %337 = sext i32 %314 to i64
  %338 = sext i32 %317 to i64
  %339 = sext i32 %320 to i64
  %340 = sext i32 %322 to i64
  %341 = sext i32 %324 to i64
  %342 = sext i32 %326 to i64
  %343 = sext i32 %328 to i64
  %344 = sext i32 %330 to i64
  %345 = sext i32 %332 to i64
  %346 = getelementptr float, ptr %81, i64 %333
  %347 = getelementptr float, ptr %85, i64 %300
  %348 = getelementptr float, ptr %81, i64 %334
  %349 = getelementptr float, ptr %85, i64 %305
  %350 = getelementptr float, ptr %81, i64 %335
  %351 = getelementptr float, ptr %85, i64 %308
  %352 = getelementptr float, ptr %87, i64 %336
  %353 = getelementptr float, ptr %288, i64 %304
  %354 = getelementptr float, ptr %289, i64 %312
  %355 = getelementptr float, ptr %290, i64 %304
  %356 = getelementptr float, ptr %291, i64 %312
  %357 = getelementptr float, ptr %87, i64 %337
  %358 = getelementptr float, ptr %85, i64 %305
  %359 = getelementptr float, ptr %85, i64 %315
  %360 = getelementptr float, ptr %85, i64 %305
  %361 = getelementptr float, ptr %85, i64 %315
  %362 = getelementptr float, ptr %87, i64 %338
  %363 = getelementptr float, ptr %85, i64 %300
  %364 = getelementptr float, ptr %85, i64 %318
  %365 = getelementptr float, ptr %85, i64 %300
  %366 = getelementptr float, ptr %85, i64 %318
  %367 = getelementptr float, ptr %92, i64 %339
  %368 = getelementptr float, ptr %85, i64 %318
  %369 = getelementptr float, ptr %92, i64 %340
  %370 = getelementptr float, ptr %85, i64 %315
  %371 = getelementptr float, ptr %92, i64 %341
  %372 = getelementptr float, ptr %85, i64 %308
  %373 = getelementptr float, ptr %97, i64 %342
  %374 = getelementptr float, ptr %81, i64 %343
  %375 = getelementptr float, ptr %85, i64 %308
  %376 = getelementptr float, ptr %101, i64 %344
  %377 = getelementptr float, ptr %104, i64 %345
  br i1 %68, label %725, label %378

378:                                              ; preds = %292
  %379 = mul i32 %103, %293
  %380 = add i32 %250, %379
  %381 = sext i32 %380 to i64
  %382 = shl nsw i64 %381, 2
  %383 = getelementptr i8, ptr %193, i64 %382
  %384 = getelementptr i8, ptr %189, i64 %382
  %385 = mul i32 %99, %293
  %386 = add i32 %248, %385
  %387 = sext i32 %386 to i64
  %388 = shl nsw i64 %387, 2
  %389 = getelementptr i8, ptr %188, i64 %388
  %390 = getelementptr i8, ptr %184, i64 %388
  %391 = mul i32 %95, %293
  %392 = add i32 %246, %391
  %393 = sext i32 %392 to i64
  %394 = shl nsw i64 %393, 2
  %395 = getelementptr i8, ptr %183, i64 %394
  %396 = getelementptr i8, ptr %180, i64 %394
  %397 = add i32 %245, %391
  %398 = sext i32 %397 to i64
  %399 = shl nsw i64 %398, 2
  %400 = getelementptr i8, ptr %179, i64 %399
  %401 = getelementptr i8, ptr %174, i64 %399
  %402 = add i32 %244, %391
  %403 = sext i32 %402 to i64
  %404 = shl nsw i64 %403, 2
  %405 = getelementptr i8, ptr %173, i64 %404
  %406 = getelementptr i8, ptr %167, i64 %404
  %407 = mul i32 %90, %293
  %408 = add i32 %242, %407
  %409 = sext i32 %408 to i64
  %410 = shl nsw i64 %409, 2
  %411 = getelementptr i8, ptr %166, i64 %410
  %412 = getelementptr i8, ptr %163, i64 %410
  %413 = add i32 %241, %407
  %414 = sext i32 %413 to i64
  %415 = shl nsw i64 %414, 2
  %416 = getelementptr i8, ptr %162, i64 %415
  %417 = getelementptr i8, ptr %157, i64 %415
  %418 = add i32 %240, %407
  %419 = sext i32 %418 to i64
  %420 = shl nsw i64 %419, 2
  %421 = getelementptr i8, ptr %156, i64 %420
  %422 = getelementptr i8, ptr %150, i64 %420
  %423 = mul i32 %84, %293
  %424 = add i32 %204, %423
  %425 = sext i32 %424 to i64
  %426 = shl nsw i64 %425, 2
  %427 = getelementptr i8, ptr %134, i64 %426
  %428 = getelementptr i8, ptr %131, i64 %426
  %429 = add i32 %203, %423
  %430 = sext i32 %429 to i64
  %431 = shl nsw i64 %430, 2
  %432 = getelementptr i8, ptr %130, i64 %431
  %433 = getelementptr i8, ptr %125, i64 %431
  %434 = add i32 %202, %423
  %435 = sext i32 %434 to i64
  %436 = shl nsw i64 %435, 2
  %437 = getelementptr i8, ptr %124, i64 %436
  %438 = getelementptr i8, ptr %119, i64 %436
  %439 = add i32 %201, %423
  %440 = sext i32 %439 to i64
  %441 = shl nsw i64 %440, 2
  %442 = getelementptr i8, ptr %118, i64 %441
  %443 = getelementptr i8, ptr %112, i64 %441
  %444 = mul i32 %106, %293
  %445 = add i32 %199, %444
  %446 = sext i32 %445 to i64
  %447 = shl nsw i64 %446, 2
  %448 = getelementptr i8, ptr %111, i64 %447
  %449 = getelementptr i8, ptr %107, i64 %447
  %450 = icmp ult ptr %449, %442
  %451 = icmp ult ptr %443, %448
  %452 = and i1 %450, %451
  %453 = icmp ult ptr %449, %437
  %454 = icmp ult ptr %438, %448
  %455 = and i1 %453, %454
  %456 = or i1 %452, %455
  %457 = icmp ult ptr %449, %432
  %458 = icmp ult ptr %433, %448
  %459 = and i1 %457, %458
  %460 = or i1 %456, %459
  %461 = icmp ult ptr %449, %427
  %462 = icmp ult ptr %428, %448
  %463 = and i1 %461, %462
  %464 = or i1 %460, %463
  %465 = icmp ult ptr %449, %211
  %466 = icmp ult ptr %210, %448
  %467 = and i1 %465, %466
  %468 = or i1 %464, %467
  %469 = icmp ult ptr %449, %214
  %470 = icmp ult ptr %213, %448
  %471 = and i1 %469, %470
  %472 = or i1 %468, %471
  %473 = icmp ult ptr %449, %219
  %474 = icmp ult ptr %218, %448
  %475 = and i1 %473, %474
  %476 = or i1 %472, %475
  %477 = icmp ult ptr %449, %226
  %478 = icmp ult ptr %225, %448
  %479 = and i1 %477, %478
  %480 = or i1 %476, %479
  %481 = icmp ult ptr %449, %228
  %482 = icmp ult ptr %227, %448
  %483 = and i1 %481, %482
  %484 = or i1 %480, %483
  %485 = icmp ult ptr %449, %231
  %486 = icmp ult ptr %230, %448
  %487 = and i1 %485, %486
  %488 = or i1 %484, %487
  %489 = icmp ult ptr %449, %233
  %490 = icmp ult ptr %232, %448
  %491 = and i1 %489, %490
  %492 = or i1 %488, %491
  %493 = icmp ult ptr %449, %236
  %494 = icmp ult ptr %235, %448
  %495 = and i1 %493, %494
  %496 = or i1 %492, %495
  %497 = icmp ult ptr %449, %238
  %498 = icmp ult ptr %237, %448
  %499 = and i1 %497, %498
  %500 = or i1 %496, %499
  %501 = icmp ult ptr %449, %421
  %502 = icmp ult ptr %422, %448
  %503 = and i1 %501, %502
  %504 = or i1 %500, %503
  %505 = icmp ult ptr %449, %416
  %506 = icmp ult ptr %417, %448
  %507 = and i1 %505, %506
  %508 = or i1 %504, %507
  %509 = icmp ult ptr %449, %411
  %510 = icmp ult ptr %412, %448
  %511 = and i1 %509, %510
  %512 = or i1 %508, %511
  %513 = icmp ult ptr %449, %405
  %514 = icmp ult ptr %406, %448
  %515 = and i1 %513, %514
  %516 = or i1 %512, %515
  %517 = icmp ult ptr %449, %400
  %518 = icmp ult ptr %401, %448
  %519 = and i1 %517, %518
  %520 = or i1 %516, %519
  %521 = icmp ult ptr %449, %395
  %522 = icmp ult ptr %396, %448
  %523 = and i1 %521, %522
  %524 = or i1 %520, %523
  %525 = icmp ult ptr %449, %389
  %526 = icmp ult ptr %390, %448
  %527 = and i1 %525, %526
  %528 = or i1 %524, %527
  %529 = icmp ult ptr %449, %383
  %530 = icmp ult ptr %384, %448
  %531 = and i1 %529, %530
  %532 = or i1 %528, %531
  %533 = icmp ult ptr %449, getelementptr inbounds nuw (i8, ptr @omega, i64 4)
  %534 = icmp ugt ptr %448, @omega
  %535 = and i1 %533, %534
  %536 = or i1 %532, %535
  br i1 %536, label %725, label %537

537:                                              ; preds = %378
  %538 = load float, ptr @omega, align 4, !tbaa !14, !alias.scope !31
  %539 = insertelement <4 x float> poison, float %538, i64 0
  %540 = shufflevector <4 x float> %539, <4 x float> poison, <4 x i32> zeroinitializer
  br label %541

541:                                              ; preds = %541, %537
  %542 = phi i64 [ 0, %537 ], [ %722, %541 ]
  %543 = phi float [ %295, %537 ], [ %721, %541 ]
  %544 = or disjoint i64 %542, 1
  %545 = getelementptr float, ptr %346, i64 %544
  %546 = getelementptr i8, ptr %545, i64 16
  %547 = load <4 x float>, ptr %545, align 4, !tbaa !14, !alias.scope !34
  %548 = load <4 x float>, ptr %546, align 4, !tbaa !14, !alias.scope !34
  %549 = getelementptr float, ptr %347, i64 %544
  %550 = getelementptr i8, ptr %549, i64 16
  %551 = load <4 x float>, ptr %549, align 4, !tbaa !14, !alias.scope !36
  %552 = load <4 x float>, ptr %550, align 4, !tbaa !14, !alias.scope !36
  %553 = getelementptr float, ptr %348, i64 %544
  %554 = getelementptr i8, ptr %553, i64 16
  %555 = load <4 x float>, ptr %553, align 4, !tbaa !14, !alias.scope !38
  %556 = load <4 x float>, ptr %554, align 4, !tbaa !14, !alias.scope !38
  %557 = getelementptr float, ptr %349, i64 %544
  %558 = getelementptr i8, ptr %557, i64 16
  %559 = load <4 x float>, ptr %557, align 4, !tbaa !14, !alias.scope !40
  %560 = load <4 x float>, ptr %558, align 4, !tbaa !14, !alias.scope !40
  %561 = fmul <4 x float> %555, %559
  %562 = fmul <4 x float> %556, %560
  %563 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %547, <4 x float> %551, <4 x float> %561)
  %564 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %548, <4 x float> %552, <4 x float> %562)
  %565 = getelementptr float, ptr %350, i64 %544
  %566 = getelementptr i8, ptr %565, i64 16
  %567 = load <4 x float>, ptr %565, align 4, !tbaa !14, !alias.scope !42
  %568 = load <4 x float>, ptr %566, align 4, !tbaa !14, !alias.scope !42
  %569 = or disjoint i64 %542, 2
  %570 = getelementptr float, ptr %351, i64 %569
  %571 = getelementptr i8, ptr %570, i64 16
  %572 = load <4 x float>, ptr %570, align 4, !tbaa !14, !alias.scope !44
  %573 = load <4 x float>, ptr %571, align 4, !tbaa !14, !alias.scope !44
  %574 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %567, <4 x float> %572, <4 x float> %563)
  %575 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %568, <4 x float> %573, <4 x float> %564)
  %576 = getelementptr float, ptr %352, i64 %544
  %577 = getelementptr i8, ptr %576, i64 16
  %578 = load <4 x float>, ptr %576, align 4, !tbaa !14, !alias.scope !46
  %579 = load <4 x float>, ptr %577, align 4, !tbaa !14, !alias.scope !46
  %580 = getelementptr float, ptr %353, i64 %544
  %581 = getelementptr i8, ptr %580, i64 16
  %582 = load <4 x float>, ptr %580, align 4, !tbaa !14, !alias.scope !48
  %583 = load <4 x float>, ptr %581, align 4, !tbaa !14, !alias.scope !48
  %584 = getelementptr float, ptr %354, i64 %544
  %585 = getelementptr i8, ptr %584, i64 16
  %586 = load <4 x float>, ptr %584, align 4, !tbaa !14, !alias.scope !50
  %587 = load <4 x float>, ptr %585, align 4, !tbaa !14, !alias.scope !50
  %588 = fsub <4 x float> %582, %586
  %589 = fsub <4 x float> %583, %587
  %590 = getelementptr float, ptr %355, i64 %544
  %591 = getelementptr i8, ptr %590, i64 16
  %592 = load <4 x float>, ptr %590, align 4, !tbaa !14, !alias.scope !52
  %593 = load <4 x float>, ptr %591, align 4, !tbaa !14, !alias.scope !52
  %594 = fsub <4 x float> %588, %592
  %595 = fsub <4 x float> %589, %593
  %596 = getelementptr float, ptr %356, i64 %544
  %597 = getelementptr i8, ptr %596, i64 16
  %598 = load <4 x float>, ptr %596, align 4, !tbaa !14, !alias.scope !54
  %599 = load <4 x float>, ptr %597, align 4, !tbaa !14, !alias.scope !54
  %600 = fadd <4 x float> %594, %598
  %601 = fadd <4 x float> %595, %599
  %602 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %578, <4 x float> %600, <4 x float> %574)
  %603 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %579, <4 x float> %601, <4 x float> %575)
  %604 = getelementptr float, ptr %357, i64 %544
  %605 = getelementptr i8, ptr %604, i64 16
  %606 = load <4 x float>, ptr %604, align 4, !tbaa !14, !alias.scope !56
  %607 = load <4 x float>, ptr %605, align 4, !tbaa !14, !alias.scope !56
  %608 = getelementptr float, ptr %358, i64 %569
  %609 = getelementptr i8, ptr %608, i64 16
  %610 = load <4 x float>, ptr %608, align 4, !tbaa !14, !alias.scope !40
  %611 = load <4 x float>, ptr %609, align 4, !tbaa !14, !alias.scope !40
  %612 = getelementptr float, ptr %359, i64 %569
  %613 = getelementptr i8, ptr %612, i64 16
  %614 = load <4 x float>, ptr %612, align 4, !tbaa !14, !alias.scope !58
  %615 = load <4 x float>, ptr %613, align 4, !tbaa !14, !alias.scope !58
  %616 = fsub <4 x float> %610, %614
  %617 = fsub <4 x float> %611, %615
  %618 = getelementptr float, ptr %360, i64 %542
  %619 = getelementptr i8, ptr %618, i64 16
  %620 = load <4 x float>, ptr %618, align 4, !tbaa !14, !alias.scope !40
  %621 = load <4 x float>, ptr %619, align 4, !tbaa !14, !alias.scope !40
  %622 = fsub <4 x float> %616, %620
  %623 = fsub <4 x float> %617, %621
  %624 = getelementptr float, ptr %361, i64 %542
  %625 = getelementptr i8, ptr %624, i64 16
  %626 = load <4 x float>, ptr %624, align 4, !tbaa !14, !alias.scope !58
  %627 = load <4 x float>, ptr %625, align 4, !tbaa !14, !alias.scope !58
  %628 = fadd <4 x float> %622, %626
  %629 = fadd <4 x float> %623, %627
  %630 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %606, <4 x float> %628, <4 x float> %602)
  %631 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %607, <4 x float> %629, <4 x float> %603)
  %632 = getelementptr float, ptr %362, i64 %544
  %633 = getelementptr i8, ptr %632, i64 16
  %634 = load <4 x float>, ptr %632, align 4, !tbaa !14, !alias.scope !60
  %635 = load <4 x float>, ptr %633, align 4, !tbaa !14, !alias.scope !60
  %636 = getelementptr float, ptr %363, i64 %569
  %637 = getelementptr i8, ptr %636, i64 16
  %638 = load <4 x float>, ptr %636, align 4, !tbaa !14, !alias.scope !36
  %639 = load <4 x float>, ptr %637, align 4, !tbaa !14, !alias.scope !36
  %640 = getelementptr float, ptr %364, i64 %569
  %641 = getelementptr i8, ptr %640, i64 16
  %642 = load <4 x float>, ptr %640, align 4, !tbaa !14, !alias.scope !62
  %643 = load <4 x float>, ptr %641, align 4, !tbaa !14, !alias.scope !62
  %644 = fsub <4 x float> %638, %642
  %645 = fsub <4 x float> %639, %643
  %646 = getelementptr float, ptr %365, i64 %542
  %647 = getelementptr i8, ptr %646, i64 16
  %648 = load <4 x float>, ptr %646, align 4, !tbaa !14, !alias.scope !36
  %649 = load <4 x float>, ptr %647, align 4, !tbaa !14, !alias.scope !36
  %650 = fsub <4 x float> %644, %648
  %651 = fsub <4 x float> %645, %649
  %652 = getelementptr float, ptr %366, i64 %542
  %653 = getelementptr i8, ptr %652, i64 16
  %654 = load <4 x float>, ptr %652, align 4, !tbaa !14, !alias.scope !62
  %655 = load <4 x float>, ptr %653, align 4, !tbaa !14, !alias.scope !62
  %656 = fadd <4 x float> %650, %654
  %657 = fadd <4 x float> %651, %655
  %658 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %634, <4 x float> %656, <4 x float> %630)
  %659 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %635, <4 x float> %657, <4 x float> %631)
  %660 = getelementptr float, ptr %367, i64 %544
  %661 = getelementptr i8, ptr %660, i64 16
  %662 = load <4 x float>, ptr %660, align 4, !tbaa !14, !alias.scope !64
  %663 = load <4 x float>, ptr %661, align 4, !tbaa !14, !alias.scope !64
  %664 = getelementptr float, ptr %368, i64 %544
  %665 = getelementptr i8, ptr %664, i64 16
  %666 = load <4 x float>, ptr %664, align 4, !tbaa !14, !alias.scope !62
  %667 = load <4 x float>, ptr %665, align 4, !tbaa !14, !alias.scope !62
  %668 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %662, <4 x float> %666, <4 x float> %658)
  %669 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %663, <4 x float> %667, <4 x float> %659)
  %670 = getelementptr float, ptr %369, i64 %544
  %671 = getelementptr i8, ptr %670, i64 16
  %672 = load <4 x float>, ptr %670, align 4, !tbaa !14, !alias.scope !66
  %673 = load <4 x float>, ptr %671, align 4, !tbaa !14, !alias.scope !66
  %674 = getelementptr float, ptr %370, i64 %544
  %675 = getelementptr i8, ptr %674, i64 16
  %676 = load <4 x float>, ptr %674, align 4, !tbaa !14, !alias.scope !58
  %677 = load <4 x float>, ptr %675, align 4, !tbaa !14, !alias.scope !58
  %678 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %672, <4 x float> %676, <4 x float> %668)
  %679 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %673, <4 x float> %677, <4 x float> %669)
  %680 = getelementptr float, ptr %371, i64 %544
  %681 = getelementptr i8, ptr %680, i64 16
  %682 = load <4 x float>, ptr %680, align 4, !tbaa !14, !alias.scope !68
  %683 = load <4 x float>, ptr %681, align 4, !tbaa !14, !alias.scope !68
  %684 = getelementptr float, ptr %372, i64 %542
  %685 = getelementptr i8, ptr %684, i64 16
  %686 = load <4 x float>, ptr %684, align 4, !tbaa !14, !alias.scope !44
  %687 = load <4 x float>, ptr %685, align 4, !tbaa !14, !alias.scope !44
  %688 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %682, <4 x float> %686, <4 x float> %678)
  %689 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %683, <4 x float> %687, <4 x float> %679)
  %690 = getelementptr float, ptr %373, i64 %544
  %691 = getelementptr i8, ptr %690, i64 16
  %692 = load <4 x float>, ptr %690, align 4, !tbaa !14, !alias.scope !70
  %693 = load <4 x float>, ptr %691, align 4, !tbaa !14, !alias.scope !70
  %694 = fadd <4 x float> %688, %692
  %695 = fadd <4 x float> %689, %693
  %696 = getelementptr float, ptr %374, i64 %544
  %697 = getelementptr i8, ptr %696, i64 16
  %698 = load <4 x float>, ptr %696, align 4, !tbaa !14, !alias.scope !72
  %699 = load <4 x float>, ptr %697, align 4, !tbaa !14, !alias.scope !72
  %700 = getelementptr float, ptr %375, i64 %544
  %701 = getelementptr i8, ptr %700, i64 16
  %702 = load <4 x float>, ptr %700, align 4, !tbaa !14, !alias.scope !44
  %703 = load <4 x float>, ptr %701, align 4, !tbaa !14, !alias.scope !44
  %704 = fneg <4 x float> %702
  %705 = fneg <4 x float> %703
  %706 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %694, <4 x float> %698, <4 x float> %704)
  %707 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %695, <4 x float> %699, <4 x float> %705)
  %708 = getelementptr float, ptr %376, i64 %544
  %709 = getelementptr i8, ptr %708, i64 16
  %710 = load <4 x float>, ptr %708, align 4, !tbaa !14, !alias.scope !74
  %711 = load <4 x float>, ptr %709, align 4, !tbaa !14, !alias.scope !74
  %712 = fmul <4 x float> %706, %710
  %713 = fmul <4 x float> %707, %711
  %714 = fmul <4 x float> %712, %712
  %715 = fmul <4 x float> %713, %713
  %716 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %540, <4 x float> %712, <4 x float> %702)
  %717 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %540, <4 x float> %713, <4 x float> %703)
  %718 = getelementptr float, ptr %377, i64 %544
  %719 = getelementptr i8, ptr %718, i64 16
  store <4 x float> %716, ptr %718, align 4, !tbaa !14, !alias.scope !76, !noalias !78
  store <4 x float> %717, ptr %719, align 4, !tbaa !14, !alias.scope !76, !noalias !78
  %720 = tail call float @llvm.vector.reduce.fadd.v4f32(float %543, <4 x float> %714)
  %721 = tail call float @llvm.vector.reduce.fadd.v4f32(float %720, <4 x float> %715)
  %722 = add nuw i64 %542, 8
  %723 = icmp eq i64 %722, %69
  br i1 %723, label %724, label %541, !llvm.loop !79

724:                                              ; preds = %541
  br i1 %71, label %822, label %725

725:                                              ; preds = %378, %292, %724
  %726 = phi i64 [ 1, %378 ], [ 1, %292 ], [ %70, %724 ]
  %727 = phi float [ %295, %378 ], [ %295, %292 ], [ %721, %724 ]
  br label %728

728:                                              ; preds = %725, %728
  %729 = phi i64 [ %743, %728 ], [ %726, %725 ]
  %730 = phi float [ %817, %728 ], [ %727, %725 ]
  %731 = getelementptr float, ptr %346, i64 %729
  %732 = load float, ptr %731, align 4, !tbaa !14
  %733 = getelementptr float, ptr %347, i64 %729
  %734 = load float, ptr %733, align 4, !tbaa !14
  %735 = getelementptr float, ptr %348, i64 %729
  %736 = load float, ptr %735, align 4, !tbaa !14
  %737 = getelementptr float, ptr %349, i64 %729
  %738 = load float, ptr %737, align 4, !tbaa !14
  %739 = fmul float %736, %738
  %740 = tail call float @llvm.fmuladd.f32(float %732, float %734, float %739)
  %741 = getelementptr float, ptr %350, i64 %729
  %742 = load float, ptr %741, align 4, !tbaa !14
  %743 = add nuw nsw i64 %729, 1
  %744 = getelementptr float, ptr %351, i64 %743
  %745 = load float, ptr %744, align 4, !tbaa !14
  %746 = tail call float @llvm.fmuladd.f32(float %742, float %745, float %740)
  %747 = getelementptr float, ptr %352, i64 %729
  %748 = load float, ptr %747, align 4, !tbaa !14
  %749 = getelementptr float, ptr %353, i64 %729
  %750 = load float, ptr %749, align 4, !tbaa !14
  %751 = getelementptr float, ptr %354, i64 %729
  %752 = load float, ptr %751, align 4, !tbaa !14
  %753 = fsub float %750, %752
  %754 = getelementptr float, ptr %355, i64 %729
  %755 = load float, ptr %754, align 4, !tbaa !14
  %756 = fsub float %753, %755
  %757 = getelementptr float, ptr %356, i64 %729
  %758 = load float, ptr %757, align 4, !tbaa !14
  %759 = fadd float %756, %758
  %760 = tail call float @llvm.fmuladd.f32(float %748, float %759, float %746)
  %761 = getelementptr float, ptr %357, i64 %729
  %762 = load float, ptr %761, align 4, !tbaa !14
  %763 = getelementptr float, ptr %358, i64 %743
  %764 = load float, ptr %763, align 4, !tbaa !14
  %765 = getelementptr float, ptr %359, i64 %743
  %766 = load float, ptr %765, align 4, !tbaa !14
  %767 = fsub float %764, %766
  %768 = add nsw i64 %729, -1
  %769 = getelementptr float, ptr %360, i64 %768
  %770 = load float, ptr %769, align 4, !tbaa !14
  %771 = fsub float %767, %770
  %772 = getelementptr float, ptr %361, i64 %768
  %773 = load float, ptr %772, align 4, !tbaa !14
  %774 = fadd float %771, %773
  %775 = tail call float @llvm.fmuladd.f32(float %762, float %774, float %760)
  %776 = getelementptr float, ptr %362, i64 %729
  %777 = load float, ptr %776, align 4, !tbaa !14
  %778 = getelementptr float, ptr %363, i64 %743
  %779 = load float, ptr %778, align 4, !tbaa !14
  %780 = getelementptr float, ptr %364, i64 %743
  %781 = load float, ptr %780, align 4, !tbaa !14
  %782 = fsub float %779, %781
  %783 = getelementptr float, ptr %365, i64 %768
  %784 = load float, ptr %783, align 4, !tbaa !14
  %785 = fsub float %782, %784
  %786 = getelementptr float, ptr %366, i64 %768
  %787 = load float, ptr %786, align 4, !tbaa !14
  %788 = fadd float %785, %787
  %789 = tail call float @llvm.fmuladd.f32(float %777, float %788, float %775)
  %790 = getelementptr float, ptr %367, i64 %729
  %791 = load float, ptr %790, align 4, !tbaa !14
  %792 = getelementptr float, ptr %368, i64 %729
  %793 = load float, ptr %792, align 4, !tbaa !14
  %794 = tail call float @llvm.fmuladd.f32(float %791, float %793, float %789)
  %795 = getelementptr float, ptr %369, i64 %729
  %796 = load float, ptr %795, align 4, !tbaa !14
  %797 = getelementptr float, ptr %370, i64 %729
  %798 = load float, ptr %797, align 4, !tbaa !14
  %799 = tail call float @llvm.fmuladd.f32(float %796, float %798, float %794)
  %800 = getelementptr float, ptr %371, i64 %729
  %801 = load float, ptr %800, align 4, !tbaa !14
  %802 = getelementptr float, ptr %372, i64 %768
  %803 = load float, ptr %802, align 4, !tbaa !14
  %804 = tail call float @llvm.fmuladd.f32(float %801, float %803, float %799)
  %805 = getelementptr float, ptr %373, i64 %729
  %806 = load float, ptr %805, align 4, !tbaa !14
  %807 = fadd float %804, %806
  %808 = getelementptr float, ptr %374, i64 %729
  %809 = load float, ptr %808, align 4, !tbaa !14
  %810 = getelementptr float, ptr %375, i64 %729
  %811 = load float, ptr %810, align 4, !tbaa !14
  %812 = fneg float %811
  %813 = tail call float @llvm.fmuladd.f32(float %807, float %809, float %812)
  %814 = getelementptr float, ptr %376, i64 %729
  %815 = load float, ptr %814, align 4, !tbaa !14
  %816 = fmul float %813, %815
  %817 = tail call float @llvm.fmuladd.f32(float %816, float %816, float %730)
  %818 = load float, ptr @omega, align 4, !tbaa !14
  %819 = tail call float @llvm.fmuladd.f32(float %818, float %816, float %811)
  %820 = getelementptr float, ptr %377, i64 %729
  store float %819, ptr %820, align 4, !tbaa !14
  %821 = icmp eq i64 %743, %43
  br i1 %821, label %822, label %728, !llvm.loop !80

822:                                              ; preds = %728, %724
  %823 = phi float [ %721, %724 ], [ %817, %728 ]
  %824 = icmp eq i64 %303, %42
  %825 = add i32 %293, 1
  br i1 %824, label %826, label %292, !llvm.loop !81

826:                                              ; preds = %822
  %827 = icmp eq i64 %253, %41
  %828 = add i32 %195, 1
  br i1 %827, label %829, label %194, !llvm.loop !82

829:                                              ; preds = %826
  br i1 %73, label %905, label %830

830:                                              ; preds = %829
  %831 = load ptr, ptr %7, align 8, !tbaa !10
  %832 = ptrtoint ptr %831 to i64
  %833 = load i32, ptr %37, align 8, !tbaa !23
  %834 = load i32, ptr %38, align 4, !tbaa !24
  %835 = load ptr, ptr %4, align 8, !tbaa !10
  %836 = ptrtoint ptr %835 to i64
  %837 = add i32 %833, 1
  %838 = mul i32 %834, %837
  %839 = mul i32 %833, %834
  br label %840

840:                                              ; preds = %901, %830
  %841 = phi i32 [ %904, %901 ], [ 0, %830 ]
  %842 = phi i64 [ %902, %901 ], [ 1, %830 ]
  %843 = mul i32 %49, %841
  %844 = add i32 %48, %843
  %845 = mul i32 %839, %841
  %846 = add i32 %838, %845
  %847 = mul nuw nsw i64 %842, %40
  %848 = trunc i64 %842 to i32
  %849 = mul i32 %833, %848
  br label %850

850:                                              ; preds = %897, %840
  %851 = phi i32 [ %900, %897 ], [ 0, %840 ]
  %852 = phi i64 [ %898, %897 ], [ 1, %840 ]
  %853 = trunc nuw nsw i64 %852 to i32
  %854 = add i32 %849, %853
  %855 = mul i32 %854, %834
  %856 = add nuw nsw i64 %852, %847
  %857 = trunc nuw i64 %856 to i32
  %858 = mul i32 %10, %857
  %859 = sext i32 %855 to i64
  %860 = sext i32 %858 to i64
  %861 = getelementptr float, ptr %831, i64 %859
  %862 = getelementptr float, ptr %835, i64 %860
  br i1 %74, label %888, label %863

863:                                              ; preds = %850
  %864 = mul i32 %834, %851
  %865 = add i32 %846, %864
  %866 = sext i32 %865 to i64
  %867 = shl nsw i64 %866, 2
  %868 = mul i32 %10, %851
  %869 = add i32 %844, %868
  %870 = sext i32 %869 to i64
  %871 = shl nsw i64 %870, 2
  %872 = add i64 %871, %836
  %873 = add i64 %867, %832
  %874 = sub i64 %872, %873
  %875 = icmp ult i64 %874, 32
  br i1 %875, label %888, label %876

876:                                              ; preds = %863, %876
  %877 = phi i64 [ %885, %876 ], [ 0, %863 ]
  %878 = or disjoint i64 %877, 1
  %879 = getelementptr float, ptr %861, i64 %878
  %880 = getelementptr i8, ptr %879, i64 16
  %881 = load <4 x float>, ptr %879, align 4, !tbaa !14
  %882 = load <4 x float>, ptr %880, align 4, !tbaa !14
  %883 = getelementptr float, ptr %862, i64 %878
  %884 = getelementptr i8, ptr %883, i64 16
  store <4 x float> %881, ptr %883, align 4, !tbaa !14
  store <4 x float> %882, ptr %884, align 4, !tbaa !14
  %885 = add nuw i64 %877, 8
  %886 = icmp eq i64 %885, %75
  br i1 %886, label %887, label %876, !llvm.loop !83

887:                                              ; preds = %876
  br i1 %77, label %897, label %888

888:                                              ; preds = %863, %850, %887
  %889 = phi i64 [ 1, %863 ], [ 1, %850 ], [ %76, %887 ]
  br label %890

890:                                              ; preds = %888, %890
  %891 = phi i64 [ %895, %890 ], [ %889, %888 ]
  %892 = getelementptr float, ptr %861, i64 %891
  %893 = load float, ptr %892, align 4, !tbaa !14
  %894 = getelementptr float, ptr %862, i64 %891
  store float %893, ptr %894, align 4, !tbaa !14
  %895 = add nuw nsw i64 %891, 1
  %896 = icmp eq i64 %895, %46
  br i1 %896, label %897, label %890, !llvm.loop !84

897:                                              ; preds = %890, %887
  %898 = add nuw nsw i64 %852, 1
  %899 = icmp eq i64 %898, %45
  %900 = add i32 %851, 1
  br i1 %899, label %901, label %850, !llvm.loop !85

901:                                              ; preds = %897
  %902 = add nuw nsw i64 %842, 1
  %903 = icmp eq i64 %902, %44
  %904 = add i32 %841, 1
  br i1 %903, label %905, label %840, !llvm.loop !86

905:                                              ; preds = %901, %829, %78
  %906 = phi float [ %823, %829 ], [ 0.000000e+00, %78 ], [ %823, %901 ]
  %907 = add nuw nsw i32 %79, 1
  %908 = icmp eq i32 %907, %0
  br i1 %908, label %909, label %78, !llvm.loop !87

909:                                              ; preds = %905, %8
  %910 = phi float [ undef, %8 ], [ %906, %905 ]
  ret float %910
}

; Function Attrs: mustprogress nounwind willreturn uwtable
define dso_local void @clearMat(ptr noundef captures(none) initializes((8, 24)) %0) local_unnamed_addr #6 {
  %2 = load ptr, ptr %0, align 8, !tbaa !10
  %3 = icmp eq ptr %2, null
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @free(ptr noundef nonnull %2) #17
  br label %5

5:                                                ; preds = %4, %1
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %0, i8 0, i64 24, i1 false)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local double @fflop(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #7 {
  %4 = add nsw i32 %2, -2
  %5 = sitofp i32 %4 to double
  %6 = add nsw i32 %1, -2
  %7 = sitofp i32 %6 to double
  %8 = fmul double %7, %5
  %9 = add nsw i32 %0, -2
  %10 = sitofp i32 %9 to double
  %11 = fmul double %8, %10
  %12 = fmul double %11, 3.400000e+01
  ret double %12
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @mflops(i32 noundef %0, double noundef %1, double noundef %2) local_unnamed_addr #7 {
  %4 = fdiv double %2, %1
  %5 = fmul double %4, 0x3EB0C6F7A0B5ED8D
  %6 = sitofp i32 %0 to double
  %7 = fmul double %5, %6
  ret double %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @set_param(ptr noundef writeonly captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #8 {
  %3 = load i8, ptr %1, align 1
  switch i8 %3, label %60 [
    i8 88, label %4
    i8 120, label %12
    i8 83, label %20
    i8 115, label %24
    i8 77, label %28
    i8 109, label %32
    i8 76, label %36
    i8 108, label %40
  ]

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %6 = load i8, ptr %5, align 1
  %7 = icmp eq i8 %6, 83
  br i1 %7, label %8, label %44

8:                                                ; preds = %4
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 2
  %10 = load i8, ptr %9, align 1
  %11 = icmp eq i8 %10, 0
  br i1 %11, label %62, label %44

12:                                               ; preds = %2
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %14 = load i8, ptr %13, align 1
  %15 = icmp eq i8 %14, 115
  br i1 %15, label %16, label %52

16:                                               ; preds = %12
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 2
  %18 = load i8, ptr %17, align 1
  %19 = icmp eq i8 %18, 0
  br i1 %19, label %62, label %52

20:                                               ; preds = %2
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %22 = load i8, ptr %21, align 1
  %23 = icmp eq i8 %22, 0
  br i1 %23, label %62, label %60

24:                                               ; preds = %2
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %26 = load i8, ptr %25, align 1
  %27 = icmp eq i8 %26, 0
  br i1 %27, label %62, label %60

28:                                               ; preds = %2
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %30 = load i8, ptr %29, align 1
  %31 = icmp eq i8 %30, 0
  br i1 %31, label %62, label %60

32:                                               ; preds = %2
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %34 = load i8, ptr %33, align 1
  %35 = icmp eq i8 %34, 0
  br i1 %35, label %62, label %60

36:                                               ; preds = %2
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %38 = load i8, ptr %37, align 1
  %39 = icmp eq i8 %38, 0
  br i1 %39, label %62, label %60

40:                                               ; preds = %2
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %42 = load i8, ptr %41, align 1
  %43 = icmp eq i8 %42, 0
  br i1 %43, label %62, label %60

44:                                               ; preds = %8, %4
  %45 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %46 = load i8, ptr %45, align 1
  %47 = icmp eq i8 %46, 76
  br i1 %47, label %48, label %60

48:                                               ; preds = %44
  %49 = getelementptr inbounds nuw i8, ptr %1, i64 2
  %50 = load i8, ptr %49, align 1
  %51 = icmp eq i8 %50, 0
  br i1 %51, label %62, label %60

52:                                               ; preds = %12, %16
  %53 = getelementptr inbounds nuw i8, ptr %1, i64 1
  %54 = load i8, ptr %53, align 1
  %55 = icmp eq i8 %54, 108
  br i1 %55, label %56, label %60

56:                                               ; preds = %52
  %57 = getelementptr inbounds nuw i8, ptr %1, i64 2
  %58 = load i8, ptr %57, align 1
  %59 = icmp eq i8 %58, 0
  br i1 %59, label %62, label %60

60:                                               ; preds = %48, %44, %2, %40, %36, %32, %20, %24, %28, %52, %56
  %61 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @exit(i32 noundef 6) #18
  unreachable

62:                                               ; preds = %48, %56, %36, %40, %28, %32, %20, %24, %8, %16
  %63 = phi i32 [ 32, %16 ], [ 32, %8 ], [ 64, %24 ], [ 64, %20 ], [ 128, %32 ], [ 128, %28 ], [ 256, %40 ], [ 256, %36 ], [ 512, %56 ], [ 512, %48 ]
  %64 = phi <2 x i32> [ <i32 32, i32 64>, %16 ], [ <i32 32, i32 64>, %8 ], [ <i32 64, i32 128>, %24 ], [ <i32 64, i32 128>, %20 ], [ <i32 128, i32 256>, %32 ], [ <i32 128, i32 256>, %28 ], [ <i32 256, i32 512>, %40 ], [ <i32 256, i32 512>, %36 ], [ <i32 512, i32 1024>, %56 ], [ <i32 512, i32 1024>, %48 ]
  store i32 %63, ptr %0, align 4, !tbaa !6
  %65 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store <2 x i32> %64, ptr %65, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #9

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #10

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #11

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #12

; Function Attrs: nofree nounwind uwtable
define dso_local double @second() local_unnamed_addr #8 {
  %1 = alloca %struct.timeval, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #17
  %2 = call i32 @gettimeofday(ptr noundef nonnull %1, ptr noundef null) #17
  %3 = load i32, ptr @second.base_sec, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 0
  %5 = load i32, ptr @second.base_usec, align 4
  %6 = icmp eq i32 %5, 0
  %7 = select i1 %4, i1 %6, i1 false
  %8 = load i64, ptr %1, align 8, !tbaa !88
  br i1 %7, label %9, label %14

9:                                                ; preds = %0
  %10 = trunc i64 %8 to i32
  store i32 %10, ptr @second.base_sec, align 4, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load i64, ptr %11, align 8, !tbaa !91
  %13 = trunc i64 %12 to i32
  store i32 %13, ptr @second.base_usec, align 4, !tbaa !6
  br label %25

14:                                               ; preds = %0
  %15 = sext i32 %3 to i64
  %16 = sub nsw i64 %8, %15
  %17 = sitofp i64 %16 to double
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %19 = load i64, ptr %18, align 8, !tbaa !91
  %20 = sext i32 %5 to i64
  %21 = sub nsw i64 %19, %20
  %22 = sitofp i64 %21 to double
  %23 = fdiv double %22, 1.000000e+06
  %24 = fadd double %23, %17
  br label %25

25:                                               ; preds = %14, %9
  %26 = phi double [ 0.000000e+00, %9 ], [ %24, %14 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #17
  ret double %26
}

; Function Attrs: nofree nounwind
declare noundef i32 @gettimeofday(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #13

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #14

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #15

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>) #15

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree nounwind willreturn memory(argmem: write, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nounwind willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #13 = { nofree nounwind }
attributes #14 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #15 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #16 = { nounwind allocsize(0) }
attributes #17 = { nounwind }
attributes #18 = { cold noreturn nounwind }

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
!10 = !{!11, !12, i64 0}
!11 = !{!"Mat", !12, i64 0, !7, i64 8, !7, i64 12, !7, i64 16, !7, i64 20}
!12 = !{!"p1 float", !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !8, i64 0}
!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.mustprogress"}
!18 = distinct !{!18, !17}
!19 = distinct !{!19, !17}
!20 = distinct !{!20, !17}
!21 = !{!11, !7, i64 8}
!22 = !{!11, !7, i64 12}
!23 = !{!11, !7, i64 16}
!24 = !{!11, !7, i64 20}
!25 = distinct !{!25, !17, !26, !27}
!26 = !{!"llvm.loop.isvectorized", i32 1}
!27 = !{!"llvm.loop.unroll.runtime.disable"}
!28 = distinct !{!28, !17, !27, !26}
!29 = distinct !{!29, !17, !26, !27}
!30 = distinct !{!30, !17, !27, !26}
!31 = !{!32}
!32 = distinct !{!32, !33}
!33 = distinct !{!33, !"LVerDomain"}
!34 = !{!35}
!35 = distinct !{!35, !33}
!36 = !{!37}
!37 = distinct !{!37, !33}
!38 = !{!39}
!39 = distinct !{!39, !33}
!40 = !{!41}
!41 = distinct !{!41, !33}
!42 = !{!43}
!43 = distinct !{!43, !33}
!44 = !{!45}
!45 = distinct !{!45, !33}
!46 = !{!47}
!47 = distinct !{!47, !33}
!48 = !{!49}
!49 = distinct !{!49, !33}
!50 = !{!51}
!51 = distinct !{!51, !33}
!52 = !{!53}
!53 = distinct !{!53, !33}
!54 = !{!55}
!55 = distinct !{!55, !33}
!56 = !{!57}
!57 = distinct !{!57, !33}
!58 = !{!59}
!59 = distinct !{!59, !33}
!60 = !{!61}
!61 = distinct !{!61, !33}
!62 = !{!63}
!63 = distinct !{!63, !33}
!64 = !{!65}
!65 = distinct !{!65, !33}
!66 = !{!67}
!67 = distinct !{!67, !33}
!68 = !{!69}
!69 = distinct !{!69, !33}
!70 = !{!71}
!71 = distinct !{!71, !33}
!72 = !{!73}
!73 = distinct !{!73, !33}
!74 = !{!75}
!75 = distinct !{!75, !33}
!76 = !{!77}
!77 = distinct !{!77, !33}
!78 = !{!73, !43, !39, !35, !45, !59, !63, !37, !41, !55, !53, !51, !49, !61, !57, !47, !69, !67, !65, !71, !75, !32}
!79 = distinct !{!79, !17, !26, !27}
!80 = distinct !{!80, !17, !26}
!81 = distinct !{!81, !17}
!82 = distinct !{!82, !17}
!83 = distinct !{!83, !17, !26, !27}
!84 = distinct !{!84, !17, !26}
!85 = distinct !{!85, !17}
!86 = distinct !{!86, !17}
!87 = distinct !{!87, !17}
!88 = !{!89, !90, i64 0}
!89 = !{!"timeval", !90, i64 0, !90, i64 8}
!90 = !{!"long", !8, i64 0}
!91 = !{!89, !90, i64 8}
